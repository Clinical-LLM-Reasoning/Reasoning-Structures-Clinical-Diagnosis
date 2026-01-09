import itertools
import numpy as np
from functools import partial
from tot.models import completion

# The GPT interface is wrapped in partial form to maintain compatibility with the original framework logic.
def gpt(prompt, n=1, stop=None, model=None, temperature=0):
    outputs = []
    for _ in range(n):
        out = completion(prompt, stop=stop, model=model)
        outputs.append(out.strip())
    return outputs

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:
        if y in local_value_cache:
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]

    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)

    # If value is a list, take the average or the first value.
    if isinstance(value, list):
        value = sum(value) / len(value) if value else 0.0

    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

# Modify to call task.get_prompt(step) to implement ToT prompt

def get_samples(task, x, y, step, n_generate_sample, stop):
    prompt = task.get_prompt(x, step, y)
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)
    ys = ['']
    infos = []
    for step in range(task.steps):
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, step, args.n_generate_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))

        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys

    best_final_candidate = None
    if new_ys: # Check if the final step successfully generated candidates.
        # Based on the evaluation values ​​`values` from the final step, find the optimal candidate from the complete candidate list `new_ys`.
        best_candidate_index = np.argmax(values)
        best_final_candidate = new_ys[best_candidate_index]

    if to_print:
        print("\n" + "="*50)
        print(f"SURVIVING CANDIDATES (The Beam): {ys}")
        print(f"BEST OVERALL CANDIDATE: {best_final_candidate}")
        print("="*50 + "\n")

    return best_final_candidate, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)
    prompt = task.get_prompt(x, 0)
    ys = gpt(prompt, n=args.n_generate_sample, stop=None)
    return ys, {}

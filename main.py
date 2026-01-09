# -*- coding: utf-8 -*-
import sys
import os
import json
import argparse
from tqdm import tqdm

os.makedirs("outputs", exist_ok=True)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# from tot.methods.bfs import solve
from tot.tasks import get_task
from tot.methods import get_method
from tot.methods.evaluate_utils import evaluate


def _extract_final_output(res):
    """
    To ensure compatibility with different method returns:

    - `bfs.solve` may return a string

    - `cot.solve` / `bot.solve` return `[final_line]`
    """
    if isinstance(res, list):
        return res[-1]
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='thyroid_lab')
    parser.add_argument('--method', type=str, default='pure_llm')  # 支持 bfs、dtree、cot、bot、pure_llm
    parser.add_argument('--method_generate', type=str, default='sample')
    parser.add_argument('--method_evaluate', type=str, default='value')
    parser.add_argument('--method_select', type=str, default='greedy')
    parser.add_argument('--prompt_sample', type=str, default='standard')
    parser.add_argument('--n_generate_sample', type=int, default=3)
    parser.add_argument('--n_evaluate_sample', type=int, default=1)
    parser.add_argument('--n_select_sample', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.7)
    # parser.add_argument('--backend', type=str, default='gpt-3.5-turbo')  # Switch to LLM (Qwen3-235B-A22B-Instruct-2507/gpt-3.5-turbo)
    # parser.add_argument('--backend', type=str, default='Qwen2.5-7B-Instruct-AWQ')   #Switch Qwen/Qwen2.5-7B-Instruct-AWQ or meta-llama/Meta-Llama-3-8B-Instruct'
    parser.add_argument('--backend', type=str, default='DeepSeek-V2-16B')

    parser.add_argument('--start', type=int, default=0)

    parser.add_argument('--end', type=int, default=None)  # None means all data
    parser.add_argument('--use_text', action='store_true', help='Whether to use text_summary column in ToT prompting.')

    # === Added: Self-consistency sampling count for CoT only ===
    parser.add_argument('--n_generate_cot', type=int, default=5,
                        help='When --method cot, number of self-consistency samples')

    args = parser.parse_args()
    # args.use_text = True    # Force enable text_summary

    task = get_task(args.task, use_text=args.use_text)

    method = get_method(args.method)

    if args.end is None:
        args.end = len(task)

    # print(task.get_flag_input(1))
    # return 0

    # === Checkpoint ===
    # Remove special characters from the model name (to prevent invalid filenames).
    safe_backend = args.backend.replace(":", "_").replace("/", "_")
    if args.method == "dtree":
        ckpt_name = f"llm_dtree_{safe_backend}_{'with_text' if args.use_text else 'no_text'}_predictions.json"
    else:
        ckpt_name = f"llm_{args.method}_{safe_backend}_{'with_text' if args.use_text else 'no_text'}_predictions.json"
    ckpt_path = os.path.join(os.path.dirname(__file__), "outputs", ckpt_name)

    y_true = []
    y_pred = []

    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        try:
            all_results = []
            with open(ckpt_path, "r", encoding="utf-8") as f:
                for line in f:
                # Ignore blank lines in the file
                    if line.strip():
                        try:
                            all_results.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Warning: Unresolved line found, skipped.: {line.strip()}")
                # all_results = json.load(f)
            done_ids = {r["example_id"] for r in all_results}
            y_true = [r["y_true"] for r in all_results]
            y_pred = [r["y_pred"] for r in all_results]
            print(f"[Checkpoint loaded] {len(all_results)} examples already processed.")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Warning] Checkpoint file is corrupted: {e}. Starting from scratch.")
            all_results = []
            done_ids = set()
    else:
        all_results = []
        done_ids = set()
        print("[Checkpoint] Starting from scratch.")


    # 替换原来的 for 循环部分
    for i in tqdm(range(args.start, min(args.end, len(task))), desc="Processing patients"):
        if i in done_ids:
            continue
        subject_id = task.subject_ids[i]
        print(f"\n[Patient #{i} | Subject ID: {subject_id}]")

        x = task.get_input(i)
        print(f"\nInput:\n{x}\n")

        # Unified Invocation
        res, info = method.solve(args, task, i, True)

        final_output = _extract_final_output(res)
        print(f"\nFinal Output:\n{final_output}")
        print("Correct?", task.test_output(i, final_output))

        final_output = _extract_final_output(res)
        y_true.append(task.get_answer(i))
        y_pred.append(task.format_output(final_output))

        #save checkpoints
        tmp_results = []

        tmp_results.append({
            "example_id": i,
            "subject_id": subject_id,
            "input": x,
            "final_output": final_output,
            "steps": info.get("steps", []),
            "correct": task.test_output(i, final_output),
            "y_pred": task.format_output(final_output),
            "y_true": task.get_answer(i)
        })
        all_results.append(tmp_results[0])

        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(tmp_results[0], ensure_ascii=False) + '\n')
            # json.dump(tmp_results, f, ensure_ascii=False, indent=2)

    print("\n=== Overall Evaluation ===")
    y_true = [int(label) for label in y_true]
    y_pred = [int(label) for label in y_pred]
    evaluate(
        y_true,
        y_pred,
        model_name=safe_backend,  # or args.backend
        method_name=args.method,
        buffer_path=ckpt_path  # The function will automatically generate *_metrics.json by passing in *_predictions.json.
    )
if __name__ == '__main__':
    main()

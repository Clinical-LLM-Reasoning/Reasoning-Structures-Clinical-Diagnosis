import os
import pandas as pd
from tot.tasks.base import Task
import re
import math

class ThyroidLabTask(Task):
    def __init__(self, use_text=False):
        super().__init__()
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "thyroid", "balanced_500_with_text_cleaned.csv")
        # data_path = os.path.join(os.path.dirname(__file__), "..", "data", "thyroid", "full_with_text_cleaned.csv")
        self.df = pd.read_csv(data_path)
        self.use_text = use_text

        # cache
        self.text_info_cache = {}

        # Grouping (each row = one lab record), patients are divided by subject_id.
        self.subject_groups = self.df.groupby("subject_id", sort=False)
        self.subject_ids = list(self.subject_groups.groups.keys())

        # label
        self.answers = self.df.groupby("subject_id")["label"].first().astype(str).tolist()
        self.max_num_examples = len(self.subject_ids)

        # ToT Steps (Backward Compatible)
        self.steps = 3
        self.stops = ["\n"] * self.steps
        self.value_cache = {}

    # ================= Security/General Tools =================
    def _safe(self, v):
        """Convert NaN/None to single quotes ('') to avoid mixing 'nan' text into the prompt."""
        if v is None:
            return ""
        s = str(v)
        return "" if s.lower() == "nan" else s

    def _fmt_time(self, t):
        """To prevent pandas warnings, charttime is uniformly parsed as YYYY-MM-DD HH:MM."""
        import pandas as pd
        try:
            # Specify the date format: day/month/year
            dt = pd.to_datetime(t, format="%d/%m/%Y %H:%M:%S", errors='coerce')
            if pd.isna(dt):
                return "unknown-time"
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "unknown-time"

    def _sanitize_free_text(self, s: str) -> str:
        if not s:
            return ""
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r'(?<!\n)(Step\s*\d+:)', r'\n\1', s)  # Step before line break
        s = re.sub(r'(->\s*(HIGH|LOW|NORMAL))(?=\S)', r'\1\n', s, flags=re.IGNORECASE)  # newline after arrow
        s = s.replace('[', '(').replace(']', ')').replace('`', "'")  # Remove square brackets/backticks
        s = re.sub(r'\n{3,}', '\n\n', s)  # Compress blank lines
        return s.strip()

    # ================= Name normalization & numerical analysis =================
    def _normalize_test_name(self, raw_name: str) -> str:
        """
Map the fixed item names appearing in the dataset to standard key names.

Known enumeration (case-insensitive/space-insensitive):
        - "Thyroid Stimulating Hormone"        -> "tsh"
        - "Thyroxine (T4), Free"               -> "ft4"
        - "Triiodothyronine (T3)"              -> "t3"     # Total T3
        - "Calculated Thyroxine (T4) Index"    -> "fti"    # Free Thyroxine Index
        - "Thyroxine (T4)"                     -> "t4"     # Total T4
        - "Thyroid Peroxidase Antibodies"      -> "tpoab"
        """
        if not raw_name:
            return ""
        # Lightweight normalization: Remove leading and trailing spaces, unify lowercase, and compress extra spaces.
        key = " ".join(str(raw_name).strip().lower().split())

        mapping = {
            "thyroid stimulating hormone": "tsh",
            "thyroxine (t4), free": "ft4",
            "triiodothyronine (t3)": "t3",
            "calculated thyroxine (t4) index": "fti",
            "thyroxine (t4)": "t4",
            "thyroid peroxidase antibodies": "tpoab",
        }
        return mapping.get(key, raw_name)

    def _parse_numeric(self, v):
        """
        Attempt to parse the number; returns None if unsuccessful.
        """
        if v is None:
            return None
        s = str(v).strip()
        s = re.sub(r'[^\d\.\-eE]', '', s)
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _row_value_text(self, r) -> str:
        """
        Use `test_value` first; if it's empty, then fall back to `value`.
        """
        tv = self._safe(r.get('test_value'))
        if tv != "":
            return tv
        return self._safe(r.get('value'))

    def _row_flag_text(self, r) -> str:
        """
        Read the flags that are already present in the dataset (if any).
        """
        f = self._safe(r.get('flag')).upper()
        if f in ("HIGH", "LOW", "NORMAL"):
            return f
        return ""

    # ================= It will iterate through sessions in chronological order (one session per charttime). =================
    def _iter_sessions(self, group_df: pd.DataFrame):
        # Create a sortable time column
        sdf = group_df.copy()
        sdf['_charttime_dt'] = pd.to_datetime(sdf['charttime'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
        # First sort by time in ascending order, then by the original charttime character order to ensure stability.
        sdf = sdf.sort_values(by=['_charttime_dt', 'charttime'], ascending=[True, True])

        # Order-preserving deduplication charttime value
        ordered_charttimes = list(dict.fromkeys(sdf['charttime'].tolist()))

        for ct in ordered_charttimes:
            sub = sdf[sdf['charttime'] == ct]
            ts = self._fmt_time(ct)
            yield ts, sub

    # ================== Construct the original block (by session) ==================
    def _build_lab_block(self, subject_id: str, group_df: pd.DataFrame) -> str:
        blocks = []
        for ts, sub in self._iter_sessions(group_df):
            header = f"### Test Session ({ts})"
            lines = []
            for _, r in sub.iterrows():
                test_name = self._safe(r.get('test_name'))
                test_value = self._row_value_text(r)
                unit = self._safe(r.get('unit'))
                lo = self._safe(r.get('ref_range_lower'))
                hi = self._safe(r.get('ref_range_upper'))
                if unit:
                    lines.append(f"- {test_name}: {test_value} {unit} (ref {lo} - {hi})")
                else:
                    lines.append(f"- {test_name}: {test_value} (ref {lo} - {hi})")
            block_text = header + "\n" + "\n".join(lines)
            blocks.append(block_text)

        preface = (
            f"Patient ID: {subject_id}\n"
            "The following are multiple thyroid lab test sessions for the same patient.\n"
        )
        lab_block = preface + "\n\n".join(blocks)
        return self._sanitize_free_text(lab_block)

    # ================== Flag Block Construction (Programmatic judgment, backtracking the flag column) ==================
    def _build_flag_block(self, subject_id: str, group_df: pd.DataFrame) -> str:
        blocks = []
        for ts, sub in self._iter_sessions(group_df):
            lines = []
            for _, r in sub.iterrows():
                test_name_raw = self._safe(r.get('test_name'))
                val_text = self._row_value_text(r)
                lo = self._safe(r.get('ref_range_lower'))
                hi = self._safe(r.get('ref_range_upper'))

                v = self._parse_numeric(val_text)
                vlo = self._parse_numeric(lo)
                vhi = self._parse_numeric(hi)

                # Priority calculation; if calculation fails, fall back to the data's built-in flag.
                if v is not None and vlo is not None and vhi is not None:
                    if v < vlo:
                        flag_calc = "LOW"
                    elif v > vhi:
                        flag_calc = "HIGH"
                    else:
                        flag_calc = "NORMAL"
                else:
                    flag_calc = self._row_flag_text(r) or "UNKNOWN"

                lines.append(f"{test_name_raw}: {flag_calc}")
            block_text = f"### Session ({ts})\n" + "\n".join(lines)
            blocks.append(block_text)

        preface = (
            f"Patient ID: {subject_id}\n"
            "Structured HIGH/LOW/NORMAL judgments per session.\n"
        )
        flag_block = preface + "\n\n".join(blocks)
        return self._sanitize_free_text(flag_block)

    def get_flag_input(self, idx):
        """
        Returning a flag_block (program-defined) is used to prevent LLM from performing numerical inference.
        """
        subject_id = self.subject_ids[idx]
        group = self.subject_groups.get_group(subject_id)
        flag_block = self._build_flag_block(subject_id, group)
        return {"flag_block": flag_block, "idx": idx, "subject_id": subject_id}

    # ================== Problem Distiller ==================
    def distill_problem(self, idx):
        """
        Organize the "line-by-line lab notes" into sessions by time, extract core hormones and flags, and generate an abstract.

        Output structure:
        {
            'subject_id': ...,
            'sessions': [ { 'time':..., 'labs':{norm_name: {value, lower, upper, flag, raw_name, value_text}}, 'abnormal_flags':[...]}],
            'aggregate': { norm_name: {...} }  # Keep the latest
            'summary_text': 'TSH x(HIGH)|FT4 y(NORMAL)|...',
            'abnormal_tokens': [...]
        }
        """
        subject_id = self.subject_ids[idx]
        group = self.subject_groups.get_group(subject_id)

        sessions = []
        agg_map = {}  # Latest value
        abnormal_tokens = []

        for ts, sub in self._iter_sessions(group):
            lab_dict = {}
            session_abnormals = []

            for _, r in sub.iterrows():
                raw_name = self._safe(r.get('test_name'))
                norm_name = self._normalize_test_name(raw_name)
                val_text = self._row_value_text(r)
                lo_raw = self._safe(r.get('ref_range_lower'))
                hi_raw = self._safe(r.get('ref_range_upper'))

                v = self._parse_numeric(val_text)
                vlo = self._parse_numeric(lo_raw)
                vhi = self._parse_numeric(hi_raw)

                if v is not None and vlo is not None and vhi is not None:
                    if v < vlo:
                        flag = "LOW"
                    elif v > vhi:
                        flag = "HIGH"
                    else:
                        flag = "NORMAL"
                else:
                    flag = self._row_flag_text(r) or "UNKNOWN"

                lab_dict[norm_name] = {
                    "raw_name": raw_name,
                    "value": v,
                    "value_text": val_text,
                    "lower": vlo,
                    "upper": vhi,
                    "flag": flag
                }
                # The aggregation retains the latest version (because it has been sorted in ascending order of time).
                agg_map[norm_name] = lab_dict[norm_name]
                if flag in ("HIGH", "LOW"):
                    session_abnormals.append(f"{norm_name}:{flag}")

            sessions.append({
                "time": ts,
                "labs": lab_dict,
                "abnormal_flags": session_abnormals
            })
            abnormal_tokens.extend(session_abnormals)

        # Generate summary text
        summary_segments = []
        for key in ("tsh", "ft4", "t3", "t4", "fti", "tpoab"):  # Add fti / tpoab
            if agg_map.get(key):
                tv = agg_map[key]["value"]
                tf = agg_map[key]["flag"]
                summary_segments.append(f"{key.upper()} {tv}({tf})")
        if not summary_segments:
            summary_segments.append("No core thyroid hormones parsed")
        summary_text = " | ".join(summary_segments)

        return {
            "subject_id": subject_id,
            "sessions": sessions,
            "aggregate": agg_map,
            "summary_text": summary_text,
            "abnormal_tokens": abnormal_tokens
        }

    def get_structured_summary(self, idx):
        return self.distill_problem(idx)

    # Optional: Vectorized features
    def get_lab_feature_vector(self, idx):
        summary = self.get_structured_summary(idx)
        agg = summary["aggregate"]
        def val_or_nan(key):
            return agg.get(key, {}).get("value", math.nan)
        return {
            "tsh": val_or_nan("tsh"),
            "ft4": val_or_nan("ft4"),
            "t3": val_or_nan("t3"),
            "t4": val_or_nan("t4"),
            "fti": val_or_nan("fti"),
            "tpoab": val_or_nan("tpoab"),
            "n_sessions": len(summary["sessions"]),
            "n_abnormal": sum(len(s["abnormal_flags"]) for s in summary["sessions"])
        }

    # ================== Existing interface (maintaining compatibility) ==================
    def __len__(self):
        return len(self.subject_ids)

    def get_input(self, idx):
        subject_id = self.subject_ids[idx]
        group = self.subject_groups.get_group(subject_id)

        if self.use_text:
            text_column = group["text_summary"].dropna().unique()
            if len(text_column) > 0:
                self.text_info_cache[idx] = str(text_column[0])

        lab_block = self._build_lab_block(subject_id, group)
        return {"lab_block": lab_block, "idx": idx}

    def test_output(self, idx, output):
        match = re.search(r'\b([01])\b', output)
        if match:
            predicted = match.group(1)
            return str(self.answers[idx]).strip() == predicted
        else:
            print(f"[Warning] Unable to extract 0/1 from the output：{output}")
            return False

    def format_output(self, output):
        match = re.search(r'\b([01])\b', output)
        if match:
            predicted = match.group(1)
            return int(predicted)
        else:
            print(f"[Warning] Unable to extract 0/1 from the output：{output}")
            return -1

    # ================== Value Prompt ==================
    def value_prompt_wrap(self, x, y):
        if isinstance(x, dict):
            x = x.get("lab_block", "") or x.get("flag_block", "") or x.get("summary_text", "")
        x = self._sanitize_free_text(x)
        y = self._sanitize_free_text(y)
        return (
            "You are a senior endocrinologist evaluating a junior doctor's clinical reasoning.\n\n"
            f"Patient data:\n{x}\n\n"
            f"Clinical reasoning:\n{y}\n\n"
            "Question: Is the reasoning medically reasonable?\n"
            "Reply only with 1 (reasonable) or 0 (errors)."
        )

    def value_outputs_unwrap(self, x, y, value_outputs):
        values = []
        for output in value_outputs:
            output = output.strip().lower()
            if re.match(r'^1\b', output):
                values.append(1.0)
            elif re.match(r'^0\b', output):
                values.append(0.0)
            else:
                try:
                    values.append(float(output))
                except Exception:
                    values.append(0.0)
        return values

    # ================== ToT Prompt ==================
    def get_prompt(self, x, step, y=None):
        if isinstance(x, dict):
            lab_block = x.get("lab_block", "") or x.get("flag_block", "") or x.get("summary_text", "")
            idx = x.get("idx", None)
        else:
            lab_block = x
            idx = None
        lab_block = self._sanitize_free_text(lab_block)
        if y is not None:
            y = self._sanitize_free_text(y)

        if step == 0:
            return (
                "You are a clinical assistant reviewing multiple lab test results for the same patient.\n\n"
                f"{lab_block}\n\n"
                "Step 1:\n"
                "For each test session, determine for each lab item whether it is HIGH, LOW, or NORMAL.\n"
                "Format strictly as one item per line.\n"
                "Reply for each session block. Do not make diagnosis yet."
            )
        elif step == 1:
            return (
                "Step 1 Observations:\n"
                f"{y}\n\n{lab_block}\n\n"
                "Step 2:\n"
                "State possible implications for abnormal values; if all normal, write 'No issue'."
            )
        elif step == 2:
            return (
                "Reasoning so far (Steps 1 & 2):\n"
                f"{y}\n\n{lab_block}\n\n"
                "Step 3:\n"
                "Does this patient likely have thyroid disease?\n"
                "Reply ONLY with 1 (yes) or 0 (no)."
            )
        else:
            raise ValueError("Invalid step")

    # ================== Other packaging ==================
    def standard_prompt_wrap(self, x, y=''):
        x = self._sanitize_free_text(x)
        y = self._sanitize_free_text(y)
        return f"You are a clinical reasoning assistant.\n\n{x}\n\nAnswer:{y}"

    def cot_prompt_wrap(self, x, y=''):
        x = self._sanitize_free_text(x)
        y = self._sanitize_free_text(y)
        return (
            "You are a clinical reasoning assistant. Start reasoning step by step.\n\n"
            f"Patient lab data:\n{x}\n\n"
            f"Your thoughts so far:\n{y}"
        )

    def propose_prompt_wrap(self, x, y=''):
        x = self._sanitize_free_text(x)
        y = self._sanitize_free_text(y)
        return (
            "You are a clinical assistant. Given the current reasoning below, propose next possible thoughts.\n\n"
            f"Current reasoning:\n{y}\n\n"
            f"Patient lab data:\n{x}\n\n"
            "Next thoughts (one per line):"
        )

    def vote_prompt_wrap(self, x, ys):
        x = self._sanitize_free_text(x)
        ys = [self._sanitize_free_text(c) for c in ys]
        choices = "\n".join([f"{i+1}. {cand}" for i, cand in enumerate(ys)])
        return (
            "You are an expert doctor.\n\n"
            f"Patient lab data:\n{x}\n\n"
            f"Candidate diagnoses:\n{choices}\n\n"
            "Which is the best diagnosis? Reply with the option number."
        )

    def vote_outputs_unwrap(self, outputs, n_choices):
        votes = [0] * n_choices
        for output in outputs:
            output = output.strip()
            try:
                choice = int(output)
                if 1 <= choice <= n_choices:
                    votes[choice - 1] += 1
            except Exception:
                continue
        return votes

    def get_answer(self, idx):
        return int(self.answers[idx])

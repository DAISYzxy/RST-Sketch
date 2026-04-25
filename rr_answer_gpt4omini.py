import os
import json
from typing import Iterator, Any, List, Dict
from tqdm import tqdm
import requests
import time

# =============================
# GPT-4 API (please use your own)
# =============================

url = ""
headers = {
    "Content-Type": "",
    "Authorization": ""
}

# 简洁 system 提示（避免与 PROMPT_TEMPLATE 重复）
conversation_history = [
    {"role": "system", "content": "You are a helpful and concise assistant."}
]


def get_gpt_response(query: str) -> str:
    local_history = conversation_history + [{"role": "user", "content": query}]

    data = {
        "model": "",
        "messages": local_history,
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        response_json = response.json()

        if "choices" in response_json:
            answer = response_json["choices"][0]["message"]["content"].strip()
            return answer
        else:
            print("⚠️ API 响应异常:", response_json)
            return "API_ERROR"

    except requests.exceptions.RequestException as e:
        print(f"⚠️ 请求异常: {e}")
        return "REQUEST_ERROR"
    except json.JSONDecodeError:
        print("⚠️ JSON 解析错误")
        return "JSON_ERROR"



def iter_jsonl(path: str, encoding: str = "utf-8") -> Iterator[Any]:
    with open(path, "r", encoding=encoding) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {lineno} 行不是合法 JSON：{e.msg}") from e


# =============================
# 读取数据集
# =============================

qsum: List[Any] = []
for obj in iter_jsonl("QSum_test.jsonl"):
    qsum.append(obj)

with open("claude4-5_qsum_summaries.json", "r", encoding="utf-8") as f:
    bm25_answers_list = json.load(f)

ours_candidates: List[str] = []
for item in bm25_answers_list:
    for a_item in item:
        ours_candidates.append(a_item)

with open("qa_list.json", "r", encoding="utf-8") as f:
    qsum_qa_list = json.load(f)



PROMPT_TEMPLATE = (
    "Please answer ONLY based on the given context.\n"
    "Keep the answer concise, in English, and avoid adding extra explanations.\n\n"
    "Given context:\n"
    "{retrieved_text}\n\n"
    "Question:\n"
    "{question}\n\n"
    "Answer:"
)



def generate_answer_with_gpt(prompt: str, retry: int = 3, sleep_sec: float = 1.0) -> str:
    for attempt in range(1, retry + 1):
        ans = get_gpt_response(prompt)
        if ans not in {"API_ERROR", "REQUEST_ERROR", "JSON_ERROR"}:
            return ans.strip()
        time.sleep(sleep_sec * attempt)
    return ""



def main():
    total = min(len(qsum_qa_list), len(ours_candidates))
    if total < len(qsum_qa_list) or total < len(ours_candidates):
        print(
            f"[Warning] Length mismatch: qsum_qa_list={len(qsum_qa_list)}, "
            f"ours_candidates={len(ours_candidates)}. Using total={total}."
        )

    out_jsonl = "rr_qa_answers_ours_claude.jsonl"
    out_json = "rr_qa_answers_ours_claude.json"

    aggregated: List[Dict[str, Any]] = []

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for idx in tqdm(range(total), desc="Generating (GPT-4)"):
            qa_pairs = qsum_qa_list[idx].get("qa_output", [])
            context = ours_candidates[idx]

            entry = {"index": idx, "answers": []}

            for qa_pair in qa_pairs:
                question = qa_pair["question"]
                prompt = PROMPT_TEMPLATE.format(retrieved_text=context, question=question)
                answer = generate_answer_with_gpt(prompt)

                line_obj = {"index": idx, "question": question, "answer": answer}
                fout.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

                entry["answers"].append({"question": question, "answer": answer})

            aggregated.append(entry)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_jsonl}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()


from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import time

def run_tests(model_name, output_file, trust=False, N=10, df=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not trust:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

    BASE_INSTRUCTIONS = (
        "You will be given a text written in {language}.\n"
        "The text contains words only and no punctuation.\n\n"
        "Your task is to add punctuation marks where appropriate.\n"
        "Use the following punctuation marks: . , ; : ? ! — \"\"\n"
        "Do not change the words, their order, capitalization, or spacing.\n"
        "Do not add or remove words.\n"
        "Preserve all whitespace.\n"
        "Return only the punctuated text.\n\n"
    )

    def build_few_shot_examples(df, n_examples=20):
        examples = []
        for _, row in df.iloc[:n_examples].iterrows():
            examples.append(
            "Sentence:\n"
            f"{row['sentence']}\n\n"
            "Correct punctuation:\n"
            f"{row['correct']}\n\n"
        )
        return "".join(examples)
    few_shot_block = build_few_shot_examples(df, n_examples=20)

    def query_model(sentence, lang, few_shot_block):
        prompt = (
            BASE_INSTRUCTIONS.format(language=lang)
            + "Examples:\n\n"
            + few_shot_block
            + "Now punctuate the following sentence.\n\n"
            "Sentence:\n"
            f"{sentence}\n\n"
            "Correct punctuation:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()

        del inputs, outputs

        return prompt, decoded

    records = []
    for i, row in df.iloc[20:50].iterrows():
        sent = row["sentence"]
        lang = row["language"] 
        prompt, raw_output = query_model(sent, lang, few_shot_block)
        records.append({
            "example_id": row["id"],
            "sentence": sent,
            "lang": lang,
            "prompt": prompt,
            "raw_output": raw_output,
            "correct": row.get("correct", None)
        })
        time.sleep(0.5)

    out_df = pd.DataFrame(records)
    out_df.to_excel(output_file, index=False)
    print("Done.")


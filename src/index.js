import { AutoTokenizer, AutoModelForSeq2SeqLM, env } from "@xenova/transformers";

env.localModelPath = "./models";
env.allowRemoteModels = false;

async function main() {
  const tokenizer = await AutoTokenizer.from_pretrained("flan-t5-base");
  const model = await AutoModelForSeq2SeqLM.from_pretrained("flan-t5-base");

  // Encode input
  const inputs = await tokenizer("Write a landing page for a handmade ceramic mug", {
    return_tensors: true,
  });

  // Generate manually
  const output_ids = await model.generate(inputs.input_ids, {
    max_new_tokens: 200,
    num_beams: 4,
    no_repeat_ngram_size: 3,
    repetition_penalty: 2.0,
    temperature: 0.8,
    top_p: 0.95,
    early_stopping: true,
  });

  const decoded = tokenizer.decode(output_ids[0], { skip_special_tokens: true });
  console.log("Generated:", decoded);
}

main().catch(console.error);

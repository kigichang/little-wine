use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use macross::{AutoModel, AutoTokenizer};
use tokenizers::Tokenizer;
mod token_output_stream;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};

const MODLE_NAME: &str = "Qwen/Qwen2.5-1.5B-Instruct";

pub fn generate(input: &str, device: &Device, max_tokens: usize, show_prompt: bool) -> Result<()> {
    let tokenizer = load_qwen_tokenizer()?;
    let model = Qwen::from_pretrained(MODLE_NAME, DType::F32, device)?;
    let mut pipeline = TextGeneration::new(
        model,
        tokenizer.clone(),
        299792458,
        None,
        None,
        1.1,
        64,
        &device,
        show_prompt,
    );
    pipeline.run(input, max_tokens)?;
    Ok(())
}

fn load_qwen_tokenizer() -> Result<Tokenizer> {
    let t =
        AutoTokenizer::from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").map_err(anyhow::Error::msg)?;
    Ok(t.into_inner())
}

struct Qwen;

impl AutoModel for Qwen {
    type Config = Config;
    type Model = ModelForCausalLM;

    fn load(vb: candle_nn::VarBuilder, config: &Self::Config) -> candle_core::Result<Self::Model>
    where
        Self::Model: Sized,
    {
        Ok(ModelForCausalLM::new(config, vb)?)
    }
}

struct TextGeneration {
    model: ModelForCausalLM,
    device: Device,
    tokenizer: token_output_stream::TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    show_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelForCausalLM,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
        show_prompt: bool,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: token_output_stream::TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            show_prompt,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        if self.show_prompt {
            for &t in tokens.iter() {
                if let Some(t) = self.tokenizer.next_token(t)? {
                    print!("{t}")
                }
            }
            std::io::stdout().flush()?;
            println!();
        }

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };

        let eos_token2 = match self.tokenizer.get_token("<|im_end|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|im_end|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                //println!("next_token:{}", next_token);
                break;
            }
            if index > 0 && next_token == eos_token2 {
                // 加判斷 <|im_end|> 這個 token
                // 可能一開始就會收到 <|im_end|> 這個 token，因此要加上 index > 0 的判斷
                // 可能會收不到 <|endoftext|>，反而是收收到 <|im_end|> 這個 token。
                // 依設定檔來看，應該是兩個都要判斷。
                // https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/generation_config.json
                // "eos_token_id": [
                //   151645,
                //   151643
                // ],
                //println!("next_token:{}", next_token);
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use macross::{AutoModel, AutoTokenizer};
use tokenizers::{self, Tokenizer};
mod token_output_stream;

fn main() -> Result<()> {
    let device = macross::device(false)?;
    println!("device: {:?}", device);

    let args = std::env::args().collect::<Vec<_>>();
    // println!("{:?}", args);

    let question = if args.len() > 1 {
        args[1].clone()
    } else {
        "推薦適合牛排的葡萄酒？".to_owned()
    };

    println!("loading data...");
    let reader = std::fs::File::open("clean_wine.json")?;
    let lst: Vec<data::Wine> = serde_json::from_reader(reader)?;
    let data: Vec<String> = lst
        .iter()
        .map(|wine| {
            // println!("{}", serde_json::to_string(&wine.profile)?);
            // let mut buf = Vec::new();
            // serde_json::to_writer(&mut buf, &wine.profile)
            //     .map(move |_| String::from_utf8_lossy(&buf).to_string())
            //     .map_err(E::msg)
            serde_json::to_string(&wine.profile).map_err(E::msg)
        })
        .collect::<Result<Vec<String>>>()?;

    const MODLE_NAME: &str = "BAAI/bge-m3";

    let similiarity = {
        println!("loading tokenizer...");
        let tokenizer = {
            let mut tokenizer =
                AutoTokenizer::from_pretrained(MODLE_NAME).map_err(anyhow::Error::msg)?;
            let params = tokenizers::PaddingParams::default();
            //println!("padding: {:?}", params);
            let truncation = tokenizers::TruncationParams::default();
            //println!("truncate: {:?}", truncation);
            let tokenizer = tokenizer.with_padding(Some(params));
            let tokenizer = tokenizer
                .with_truncation(Some(truncation))
                .map_err(anyhow::Error::msg)?;
            tokenizers::Tokenizer::from(tokenizer.to_owned())
        };

        println!("loading model...");
        let model = macross::models::xlm_roberta::XLMRobertaModel::from_pretrained(
            (MODLE_NAME, true),
            candle_core::DType::F32,
            &device,
        )?;

        let question_embedding = {
            let encoded_input = tokenizer.encode(question.clone(), true).map_err(E::msg)?;
            let ids = Tensor::new(encoded_input.get_ids(), &device)?;
            let ids = ids.unsqueeze(0)?;
            let type_ids = Tensor::new(encoded_input.get_type_ids(), &device)?;
            let type_ids = type_ids.unsqueeze(0)?;
            let attention_masks = Tensor::new(encoded_input.get_attention_mask(), &device)?;
            let attention_masks = attention_masks.unsqueeze(0)?;
            let embeddings = model.forward(&ids, &type_ids, &attention_masks)?;
            macross::normalize(&embeddings.i((.., 0))?.contiguous()?)?
        };
        println!("question embedding:{:?}", question_embedding.shape());

        println!("encoding data...");
        let encoded_inputs = tokenizer.encode_batch(data.clone(), true).map_err(E::msg)?;
        let ids = encoded_inputs
            .iter()
            .map(|e| Tensor::new(e.get_ids(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let ids = Tensor::stack(&ids, 0)?;

        let type_ids = encoded_inputs
            .iter()
            .map(|e| Tensor::new(e.get_type_ids(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let type_ids = Tensor::stack(&type_ids, 0)?;

        let attention_masks = encoded_inputs
            .iter()
            .map(|e| Tensor::new(e.get_attention_mask(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let attention_masks = Tensor::stack(&attention_masks, 0)?;

        // println!("data embeddings...");
        let embeddings = model.forward(&ids, &type_ids, &attention_masks)?;
        let embeddings = embeddings.i((.., 0))?.contiguous()?;
        let embeddings = macross::normalize(&embeddings)?;
        //macross::print_tensor::<f32>(&embeddings)?;
        println!("data embeddings:{:?}", embeddings.shape());

        // println!("calculating similarity...");
        let similiarity = embeddings.broadcast_matmul(&question_embedding.t()?)?;
        println!("similarity:{:?}", similiarity.shape());
        println!("calc similarity ...");
        let similiarity = similiarity.to_vec2::<f32>()?;
        let mut similiarity = similiarity
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s[0]))
            .collect::<Vec<_>>();
        // println!("similarity:{:?}", similiarity);
        similiarity.sort_by(|a, b| b.1.total_cmp(&a.1));
        // println!("similarity:{:?}", similiarity);
        similiarity
    };

    let top_k = similiarity
        .iter()
        .take(5)
        .map(|(i, s)| {
            println!("score: {s}, idx: {i}, name: {}", lst[*i].profile.name);
            lst[*i].profile.clone()
        })
        .collect::<Vec<_>>();

    // let mut buf = Vec::new();
    // serde_json::to_writer(&mut buf, &top_k)?;
    let prompt = serde_json::to_string(&top_k)?;
    let prompt = format!(
        "請使用以下葡萄酒資料：『{}』，撰寫一段有關推薦『{}』的文章，且在文章最後，必須加入以下警語：『禁止酒駕！飲酒過量，有害健康！未成年請勿飲酒！』",
        prompt,
        question
    );
    println!("問:{}", question);
    //println!("{}", prompt);
    //let prompt = r#"你是位葡萄酒愛好者，我會提供酒友的問題，以及推薦的葡萄酒資料。請你依推薦的葡萄酒資料，撰寫一段評論。問題是：「推薦適合牛排的葡萄酒？」。以下是推薦的葡萄酒的 JSON 資料：[{"酒名":"Beringer Founders' Estate Cabernet Sauvignon","簡介":"甜美的美國入門搭餐酒","參考價格":549,"產區":"California","國家":"美國","酒莊(廠)":"Beringer","葡萄品種":"Cabernet Sauvignon(100%)","購買通路":"好市多","年份":"2021年","酒精濃度":0.138,"甜度":2,"酸度":3,"飽滿度":5,"適合搭配食物":"牛排","品酒筆記":"帶有黑棗、黑李與蜜餞風味，口感甜美，單寧柔順的酒款。適合搭配牛排，也很適合很討厭酸澀葡萄酒的初學者。3.5顆星。"},{"酒名":"Motif Cabernet Sauvignon red hills","簡 介":"甜美柔滑的美國酒","參考價格":299,"產區":"Red hills lake county AVA","國家":"美國","酒莊(廠)":"Motif","葡萄品種":"Cabernet Sauvignon(100%)","購買通路":"好市多","年份":"2022年","酒精濃度":0.145,"甜度":1,"酸度":4,"飽滿度":5,"適合搭配食 物":"牛排","品酒筆記":"來自於美國 Red hills lake county AVA 產區的卡本內蘇維翁，帶有甜美的黑李子蜜餞風味，口感柔滑，單寧柔順，談不上什麼複雜度，就是款開瓶就好喝，又不會過於甜膩的日常餐酒，適合搭配牛排也很適合新手的入門酒款。4顆星。"},{"酒名":"Ségla 二軍紅酒","簡介":"細緻優雅的波爾多二軍紅酒","參考價格":2000,"產區":"Margaux","國家":"法國","酒莊(廠)":"Chateau Rauzan-Ségla","葡萄品種":"波爾多混釀(100%)","購買通路":"法蘭絲","年份":"2017年","酒精濃度":0.135,"甜度":1,"酸度":4,"飽滿度":5,"適合搭配食物":"牛排","品酒筆記":"波爾多二級酒莊 Segla 二軍紅酒，帶有明顯的木桶香氣混合黑色水果香氣，口感優雅，單寧適中搭配肋眼、紐約客牛排相當不錯。"},{"酒名":"Wynns Black Label Cabernet Sauvignon","簡介":"飽滿圓滑有層次的澳洲 卡本內蘇維翁","參考價格":1800,"產區":"Coonawarra","國家":"澳洲","酒莊(廠)":"Wynn","葡萄品種":"Cabernet Sauvignon(100%)","購買通路":"Wine O'Clock,LOCA Co","年份":"2019年","酒精濃度":0.138,"甜度":1,"酸度":4,"飽滿度":5,"適合搭配食物":"牛排","品酒筆記":"嚴選酒莊位於庫納瓦拉產區terra rossa紅土葡萄園中，20%-25%最頂級的卡本內蘇維濃 (Cabernet Sauvignon)葡萄釀造。在法國新(30%)及舊橡木桶中陳放16個月。帶有黑色水果與木桶風味，口感飽滿圓滑，單寧適中，適合搭配牛排。"},{"酒名":"Wente Merlot","簡介":"適合牛排或聚餐的美國Wente Merlot 紅酒","參考價格":619,"產區":"Central coast","國家":"美國","酒莊(廠)":"Wente","葡萄品種":"Ｍerlot(100%)","購買通路":"橡木桶洋酒","年份":"2021年","酒精濃度":0.135,"甜度":1,"酸度":4,"飽滿度":5,"適合搭配食物":"牛排","品酒筆記":"Wente 也是橡木桶架上的長年品項，自己也是喝很多次。\n\n小資對於加州 Merlot的印象很好，不會太柔軟，很有自己的風格。 這款來自於 加州 Central coast 的 Merlot，帶有黑莓、可可以及適切的桶位，剛剛好的單寧，有架構不會過澀， 這種算是中價位，適合帶去聚餐、吃牛排，也是我去橡木桶看到價錢優惠，就會毫不思索結帳的酒款，建議醒酒30分鐘。4顆星推薦。"}]"#;
    let msg = format!("<|im_start|>system\n你是位葡萄酒愛好者，你會依照酒友的問題，推薦合適的葡萄酒。<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant", prompt);
    println!("產生答案中...");
    generate(&msg, &device)?;
    Ok(())
}

fn load_qwen_tokenizer() -> Result<Tokenizer> {
    let t =
        AutoTokenizer::from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").map_err(anyhow::Error::msg)?;
    Ok(t.into_inner())
}

fn generate(input: &str, device: &Device) -> Result<()> {
    let tokenizer = load_qwen_tokenizer()?;
    let model = Qwen::from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", DType::F32, device)?;
    let mut pipeline = TextGeneration::new(
        model,
        tokenizer.clone(),
        299792458,
        None,
        None,
        1.1,
        64,
        &device,
    );
    pipeline.run(input, 10000)?;
    Ok(())
}

use candle_transformers::models::qwen2::ModelForCausalLM;
struct Qwen(ModelForCausalLM);

impl AutoModel for Qwen {
    type Config = candle_transformers::models::qwen2::Config;
    type Model = Qwen;

    fn load(vb: candle_nn::VarBuilder, config: &Self::Config) -> candle_core::Result<Self::Model>
    where
        Self::Model: Sized,
    {
        Ok(Qwen(ModelForCausalLM::new(config, vb)?))
    }
}

struct TextGeneration {
    model: Qwen,
    device: Device,
    tokenizer: token_output_stream::TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Qwen,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: token_output_stream::TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
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
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;
        println!();

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
            let logits = self.model.0.forward(&input, start_pos)?;
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

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp};
use clap::Parser;
use macross::AutoModel;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    master_key: String,
    #[arg(long, default_value = "推薦適合牛排的葡萄酒？")]
    prompt: String,
    #[arg(long, default_value_t = false)]
    show_prompt: bool,
    #[arg(long, default_value_t = 5)]
    top_k: usize,
    #[arg(long, default_value_t = 10240)]
    max_token: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let device = macross::device(false)?;
    println!("device: {:?}", device);

    let args = Args::parse();

    let question_embedding = {
        const MODLE_NAME: &str = "BAAI/bge-m3";
        println!("loading tokenizer...");
        let tokenizer = mytokenizers::tokenizer(MODLE_NAME)?;

        println!("loading model...");
        let model = macross::models::xlm_roberta::XLMRobertaModel::from_pretrained(
            (MODLE_NAME, true),
            candle_core::DType::F32,
            &device,
        )?;

        // let model =
        //     mymodel::AutoXLMRobertaModel::from_pretrained((MODLE_NAME, true), DType::F32, &device)?;

        let (ids, type_ids, attention_masks) =
            mytokenizers::encode_batch(&tokenizer, vec![args.prompt.clone()], &device)?;
        let question_embedding = model.forward(&ids, &type_ids, &attention_masks)?;
        // let question_embedding =
        //     model.forward(&ids, &attention_masks, &type_ids, None, None, None)?;
        let question_embedding = question_embedding.i((.., 0))?.contiguous()?;
        let question_embedding = macross::normalize(&question_embedding)?;
        println!("question embedding:{:?}", question_embedding.shape());
        question_embedding.to_vec2::<f32>()?
    };

    println!("searching data...");
    let wines = search_wine(&args.master_key, &question_embedding[0], args.top_k).await?;

    println!("generating answer...");
    generate(&args.prompt, &wines, &device, &args)?;
    Ok(())
}

async fn search_wine(
    master_key: &str,
    embedding: &[f32],
    top_k: usize,
) -> Result<Vec<data::WineProfile>> {
    let cli = reqwest::Client::new();
    let resp = cli
        .post("http://127.0.0.1:7700/indexes/bge-m3/search")
        .header("Authorization", format!("Bearer {}", master_key))
        .json(&json!({
            "showRankingScore": true,
            "hybrid": { "embedder": "embedding", "semanticRatio": 0.7 },
            "vector": embedding,
            "limit": top_k,

        }))
        .send()
        .await?;

    let results = resp.text().await?;
    // println!("{}", results);
    let results: serde_json::Value = serde_json::from_str(&results)?;
    let mut raws = vec![];
    for hit in results["hits"].as_array().ok_or(E::msg("no hits"))? {
        println!(
            "{:>3} {:.6} {:>4} {} {}",
            hit["id"].as_u64().unwrap(),
            hit["_rankingScore"].as_f64().unwrap(),
            hit["price"].as_f64().unwrap(),
            hit["name"].as_str().unwrap(),
            hit["note"].as_str().unwrap()
        );

        let profile: data::WineProfile =
            serde_json::from_str(hit["raw"].as_str().ok_or(E::msg("no raw"))?)?;
        raws.push(profile);
    }

    Ok(raws)
}

// -----------------------------------------------------------------------------

fn generate(
    question: &str,
    data: &[data::WineProfile],
    device: &Device,
    args: &Args,
) -> Result<()> {
    let prompt = serde_json::to_string(data)?;
    let prompt = format!(
        "請使用以下葡萄酒資料：『{}』，撰寫一段有關推薦『{}』的文章，且在文章最後，必須加入以下警語：『禁止酒駕！飲酒過量，有害健康！未成年請勿飲酒！』",
        prompt,
        question
    );
    println!("問：{}", question);
    let msg = format!("<|im_start|>system\n你是位葡萄酒愛好者，你會依照酒友的問題，推薦合適的葡萄酒。<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant", prompt);
    println!("產生答案中...");
    qwen::generate(&msg, &device, args.max_token, args.show_prompt)?;
    Ok(())
}

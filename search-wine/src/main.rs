use anyhow::{Error as E, Result};
use candle_core::{IndexOp, Tensor};
use macross::AutoModel;
use macross::AutoTokenizer;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let (master_key, question) = {
        let args = std::env::args().collect::<Vec<String>>();
        if args.len() < 2 {
            eprintln!("Usage: {} meilli-search-master-key", args[0]);
            std::process::exit(1);
        }
        if args.len() > 2 {
            (args[1].clone(), args[2].clone())
        } else {
            (args[1].clone(), "推薦適合牛排的葡萄酒？".to_owned())
        }
    };
    println!("問：{}", question);

    let device = macross::device(false)?;
    const MODLE_NAME: &str = "BAAI/bge-m3";

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
        embeddings.i((.., 0))?.contiguous()?
    };
    println!("question embedding:{:?}", question_embedding.shape());
    let question_embedding = question_embedding.to_vec2::<f32>()?;

    let cli = reqwest::Client::new();
    let resp = cli
        .post("http://127.0.0.1:7700/indexes/bge-m3/search")
        .header("Authorization", format!("Bearer {}", master_key))
        .json(&json!({
            "showRankingScore": true,
            "hybrid": { "embedder": "embedding", "semanticRatio": 0.7 },
            "vector": question_embedding[0],
            "limit": 5,

        }))
        .send()
        .await?;

    let results = resp.text().await?;
    println!("{}", results);
    let results: serde_json::Value = serde_json::from_str(&results)?;
    for hit in results["hits"].as_array().ok_or(E::msg("no hits"))? {
        println!("{} {}: {}", hit["_rankingScore"], hit["id"], hit["name"]);
        println!("{}", hit["raw"].as_str().ok_or(E::msg("no raw"))?);
    }
    Ok(())
}

use std::fs::File;

use anyhow::{Error as E, Result};
use candle_core::{DType, IndexOp, Tensor};
use macross::{AutoModel, AutoTokenizer};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let master_key = {
        let args = std::env::args().collect::<Vec<String>>();
        if args.len() < 2 {
            eprintln!("Usage: {} meilli-search-master-key", args[0]);
            std::process::exit(1);
        }
        args[1].clone()
    };

    let wines = File::open("clean_wine.json")?;
    let wines: Vec<data::Wine> = serde_json::from_reader(wines)?;
    let wine_profiles = wines
        .iter()
        .map(|wine| serde_json::to_string(&wine.profile).map_err(E::msg))
        .collect::<Result<Vec<String>>>()?;

    // for wine in wine_profiles {
    //     println!("{}", wine);
    // }

    let device = macross::device(false)?;
    const MODLE_NAME: &str = "BAAI/bge-m3";

    // let tokenizer: tokenizers::Tokenizer = {
    //     let mut tokenizer = AutoTokenizer::from_pretrained(MODLE_NAME)
    //         .map_err(E::msg)?
    //         .into_inner();
    //     let padding = tokenizers::PaddingParams::default();
    //     let truncation = tokenizers::TruncationParams::default();
    //     let tokenizer = tokenizer.with_padding(Some(padding));
    //     let tokenizer = tokenizer
    //         .with_truncation(Some(truncation))
    //         .map_err(E::msg)?;
    //     tokenizers::Tokenizer::from(tokenizer.to_owned())
    // };

    let tokenizer = mytokenizers::tokenizer(MODLE_NAME)?;
    let (input_ids, type_ids, attention_masks) =
        mytokenizers::encode_batch(&tokenizer, wine_profiles.clone(), &device)?;
    // let encoded_inputs = tokenizer
    //     .encode_batch(wine_profiles.clone(), true)
    //     .map_err(E::msg)?;

    // let input_ids = encoded_inputs
    //     .iter()
    //     .map(|input| Tensor::new(input.get_ids(), &device))
    //     .collect::<candle_core::Result<Vec<_>>>()?;
    // let input_ids = Tensor::stack(&input_ids, 0)?;

    // let type_ids = encoded_inputs
    //     .iter()
    //     .map(|input| Tensor::new(input.get_type_ids(), &device))
    //     .collect::<candle_core::Result<Vec<_>>>()?;
    // let type_ids = Tensor::stack(&type_ids, 0)?;

    // let attention_masks = encoded_inputs
    //     .iter()
    //     .map(|input| Tensor::new(input.get_attention_mask(), &device))
    //     .collect::<candle_core::Result<Vec<_>>>()?;
    // let attention_masks = Tensor::stack(&attention_masks, 0)?;

    let model = macross::models::xlm_roberta::XLMRobertaModel::from_pretrained(
        (MODLE_NAME, true),
        DType::F32,
        &device,
    )?;
    let embeddings = model.forward(&input_ids, &type_ids, &attention_masks)?;

    // let model =
    //     mymodel::AutoXLMRobertaModel::from_pretrained((MODLE_NAME, true), DType::F32, &device)?;
    //let embeddings = model.forward(&input_ids, &attention_masks, &type_ids, None, None, None)?;

    let embeddings = embeddings.i((.., 0))?.contiguous()?;
    let embeddings = macross::normalize(&embeddings)?;
    println!("embeddings: {:?}", embeddings.shape());

    println!("retrieving embeddings");
    let embeddings = embeddings.to_vec2::<f32>()?;

    println!("adding to meillisearch");
    let cli = reqwest::Client::new();

    for (i, embedding) in embeddings.iter().enumerate() {
        let resp = cli
            .post("http://127.0.0.1:7700/indexes/bge-m3/documents")
            .header("Authorization", format!("Bearer {}", master_key))
            .json(&json!({
                "id": i,
                "_vectors": {"embedding": embedding},
                "raw": wine_profiles[i],
                "name": wines[i].profile.name,
                "desc": wines[i].profile.desc,
                "price": wines[i].profile.price,
                "origin": wines[i].profile.origin,
                "country": wines[i].profile.country,
                "winery": wines[i].profile.winery,
                "grape": wines[i].profile.grape,
                "channel": wines[i].profile.channel,
                "year": wines[i].profile.year,
                "alcohol": wines[i].profile.alcohol,
                "sweetness": wines[i].profile.sweetness,
                "acidity": wines[i].profile.acidity,
                "fullness": wines[i].profile.fullness,
                "food": wines[i].profile.food,
                "note": wines[i].profile.note,
                "url": wines[i].url,
            }))
            .send()
            .await?;
        println!("{}/{}: {}", i, wines[i].profile.name, resp.text().await?);
    }

    println!("done");
    Ok(())
}

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use macross::{AutoTokenizer, ModelRepo};
use tokenizers::Tokenizer;

pub fn tokenizer<M: Into<ModelRepo>>(model: M) -> Result<Tokenizer> {
    let mut tokenizer = AutoTokenizer::from_pretrained(model).map_err(anyhow::Error::msg)?;
    let padding = tokenizers::PaddingParams::default();
    let truncation = tokenizers::TruncationParams::default();
    let tokenizer = tokenizer.with_padding(Some(padding));
    let tokenizer = tokenizer
        .with_truncation(Some(truncation))
        .map_err(anyhow::Error::msg)?;
    Ok(tokenizers::Tokenizer::from(tokenizer.to_owned()))
}

///! ouput: (ids, type_ids, attention_masks)
pub fn encode_batch(
    tokenizer: &Tokenizer,
    texts: Vec<String>,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let encoded_inputs = tokenizer.encode_batch(texts, true).map_err(E::msg)?;
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
    Ok((ids, type_ids, attention_masks))
}

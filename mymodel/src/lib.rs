use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaModel};
use macross::AutoModel;

pub struct AutoXLMRobertaModel;

impl AutoModel for AutoXLMRobertaModel {
    type Config = Config;
    type Model = XLMRobertaModel;

    fn load(vb: VarBuilder, config: &Self::Config) -> candle_core::Result<Self::Model> {
        XLMRobertaModel::new(config, vb)
    }
}

pub fn forward(
    model: &XLMRobertaModel,
    input: &Tensor,
    attention_mask: &Tensor,
    token_type_ids: &Tensor,
) -> candle_core::Result<Tensor> {
    model.forward(input, attention_mask, token_type_ids, None, None, None)
}

// let model = macross::models::xlm_roberta::XLMRobertaModel::from_pretrained(
//     (MODLE_NAME, true),
//     candle_core::DType::F32,
//     &device,
// )?;

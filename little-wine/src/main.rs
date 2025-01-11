use anyhow::{Error as E, Result};
use candle_core::IndexOp;
use macross::AutoModel;

fn main() -> Result<()> {
    const MODLE_NAME: &str = "BAAI/bge-m3";

    let device = macross::device(false)?;
    println!("device: {:?}", device);

    let args = std::env::args().collect::<Vec<_>>();
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
        .map(|wine| serde_json::to_string(&wine.profile).map_err(E::msg))
        .collect::<Result<Vec<String>>>()?;

    let similiarity = {
        println!("loading tokenizer...");
        let tokenizer = mytokenizers::tokenizer(MODLE_NAME)?;

        println!("loading model...");
        let model = macross::models::xlm_roberta::XLMRobertaModel::from_pretrained(
            (MODLE_NAME, true),
            candle_core::DType::F32,
            &device,
        )?;

        let question_embedding = {
            let (ids, type_ids, attention_masks) =
                mytokenizers::encode_batch(&tokenizer, vec![question.clone()], &device)?;
            let embeddings = model.forward(&ids, &type_ids, &attention_masks)?;
            macross::normalize(&embeddings.i((.., 0))?.contiguous()?)?
        };
        println!("question embedding:{:?}", question_embedding.shape());

        println!("encoding data...");
        let (ids, type_ids, attention_masks) =
            mytokenizers::encode_batch(&tokenizer, data.clone(), &device)?;
        let embeddings = model.forward(&ids, &type_ids, &attention_masks)?;
        let embeddings = embeddings.i((.., 0))?.contiguous()?;
        let embeddings = macross::normalize(&embeddings)?;
        println!("data embeddings:{:?}", embeddings.shape());

        let similiarity = embeddings.broadcast_matmul(&question_embedding.t()?)?;
        println!("similarity:{:?}", similiarity.shape());
        println!("calc similarity ...");
        let similiarity = similiarity.to_vec2::<f32>()?;
        let mut similiarity = similiarity
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s[0]))
            .collect::<Vec<_>>();
        similiarity.sort_by(|a, b| b.1.total_cmp(&a.1));
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

    let prompt = serde_json::to_string(&top_k)?;
    let prompt = format!(
        "請使用以下葡萄酒資料：『{}』，撰寫一段有關推薦『{}』的文章，且在文章最後，必須加入以下警語：『禁止酒駕！飲酒過量，有害健康！未成年請勿飲酒！』",
        prompt,
        question
    );
    println!("問：{}", question);
    let msg = format!("<|im_start|>system\n你是位葡萄酒愛好者，你會依照酒友的問題，推薦合適的葡萄酒。<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant", prompt);
    println!("產生答案中...");
    qwen::generate(&msg, &device, 10240, true)?;

    Ok(())
}

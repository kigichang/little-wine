use anyhow::Result;
use reqwest;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() != 2 {
        eprintln!("Usage: {} meilli-search-master-key", args[0]);
        std::process::exit(1);
    }
    // println!("{}", args[1]);
    // let cli = Client::new("http://127.0.0.1:7700", Some(&args[1]))?;

    // let keys = cli.get_keys().await?;
    // println!("{:?}", keys);

    // let task_info = cli.create_index("bge-m3", Some("id")).await?;
    // println!("create index");
    // println!("{:?}", task_info);

    // let index = cli.get_index("bge-m3").await?;
    // println!("get index");
    // println!("{:?}", index);

    let cli = reqwest::Client::new();
    let resp = cli
        .get("http://127.0.0.1:7700/indexes/bge-m3/settings")
        .header("Authorization", format!("Bearer {}", args[1]))
        .send()
        .await?;
    println!("{}", resp.text().await?);

    // let settings = json!(json!({
    //     "embedders": {
    //         "embedding": {
    //             "source": "userProvided",
    //             "dimensions": 1024,
    //         }
    //     }
    // }));
    // let resp = cli
    //     .patch("http://127.0.0.1:7700/indexes/bge-m3/settings")
    //     .header("Authorization", format!("Bearer {}", args[1]))
    //     .json(&settings)
    //     .send()
    //     .await?;
    // println!("{}", resp.text().await?);
    Ok(())
}

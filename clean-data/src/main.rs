use anyhow::Result;
fn main() -> Result<()> {
    let reader = std::fs::File::open("little_wine.json")?;

    let mut lst: Vec<data::Wine> = serde_json::from_reader(reader)?;

    for wine in &mut lst {
        wine.profile.name = wine.profile.name.trim().to_owned();
        wine.profile.desc = wine.profile.desc.trim().to_owned();
        wine.profile.origin = wine.profile.origin.trim().to_owned();
        wine.profile.country = wine.profile.country.trim().to_owned();
        wine.profile.winery = wine.profile.winery.trim().to_owned();
        wine.profile.grape = wine.profile.grape.trim().to_owned();
        wine.profile.channel = wine.profile.channel.trim().to_owned();
        wine.profile.year = wine.profile.year.trim().to_owned();
        wine.profile.food = wine.profile.food.trim().to_owned();
        wine.profile.note = wine.profile.note.trim().to_owned();
        wine.url = wine.url.trim().to_owned();
    }

    println!("{:?}", lst);

    let writer = std::fs::File::create("clean_wine.json")?;
    serde_json::to_writer_pretty(writer, &lst)?;

    let data: Vec<String> = lst
        .iter()
        .map(|wine| serde_json::to_string(&wine.profile).map_err(anyhow::Error::msg))
        .collect::<Result<Vec<String>>>()?;

    println!("{:?}", data);
    Ok(())
}

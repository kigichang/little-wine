use serde::Deserialize;
use serde::Serialize;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wine {
    #[serde(flatten)]
    pub profile: WineProfile,
    #[serde(rename = "URL")]
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WineProfile {
    #[serde(rename = "酒名")]
    pub name: String,

    #[serde(rename = "簡介")]
    pub desc: String,

    #[serde(rename = "參考價格")]
    pub price: u32,

    #[serde(rename = "產區")]
    pub origin: String,

    #[serde(rename = "國家")]
    pub country: String,

    #[serde(rename = "酒莊(廠)")]
    pub winery: String,

    #[serde(rename = "葡萄品種")]
    pub grape: String,

    #[serde(rename = "購買通路")]
    pub channel: String,

    #[serde(rename = "年份")]
    pub year: String,

    #[serde(rename = "酒精濃度")]
    pub alcohol: f32,

    #[serde(rename = "甜度")]
    pub sweetness: u8,

    #[serde(rename = "酸度")]
    pub acidity: u8,

    #[serde(rename = "飽滿度")]
    pub fullness: u8,

    #[serde(rename = "適合搭配食物")]
    pub food: String,

    #[serde(rename = "品酒筆記")]
    pub note: String,
}

impl std::fmt::Display for WineProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#"{{"酒名":{:#?},"簡介":{:#?},"參考價格":{},"產區":{:#?},"國家":{:#?},"酒莊(廠)":{:#?},"葡萄品種":{:#?},"購買通路":{:#?},"年份":{:#?},"酒精濃度":{:#?},"甜度":{:#?},"酸度":{:#?},"飽滿度":{:#?},"適合搭配食物":{:#?},"品酒筆記":{:#?}}}"#,
            self.name,
            self.desc,
            self.price,
            self.origin,
            self.country,
            self.winery,
            self.grape,
            self.channel,
            self.year,
            self.alcohol,
            self.sweetness,
            self.acidity,
            self.fullness,
            self.food,
            self.note,
        )
    }
}

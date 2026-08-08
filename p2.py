import pandas as pd

# ------------------------------------------------------------

df = pd.read_csv("diamonds.csv")

print(df.columns)
print(df.index)
print(df.dtypes)

print(df["cut"].unique())
print(df["cut"].value_counts())
print(df["cut"].nunique())

# ------------------------------------------------------------

print(df.loc[10])
print(df.loc[10:20, ["carat", "cut", "price"]])
print(df.iloc[10])
print(df.iloc[10:20, 0:4])
print(df[df["cut"].isin(["Ideal", "Premium"])])
print(df[df["price"].between(1000, 3000)])
print(df.query("price > 3000"))
print(df.filter(items=["carat", "cut", "color", "price"]))
#------------------------------------------------------------
df = df.sort_values("price")
df = df.sort_values("price", ascending=False)
df = df.sort_index()
df = df.rename(columns={
    "carat": "Carat",
    "cut": "Cut",
    "color": "Color",
    "clarity": "Clarity"
})

df["price"] = df["price"].astype("float64")

df["Cut"] = df["Cut"].replace({
    "Very Good": "VeryGood"
})

df["Carat"] = df["Carat"].fillna(df["Carat"].mean())

df = df.dropna()

df = df.drop_duplicates()
# -----------------------------------------------------------

df["Cut_Code"] = df["Cut"].map({
    "Ideal": 5,
    "Premium": 4,
    "VeryGood": 3,
    "Good": 2,
    "Fair": 1
})

df["Price_Double"] = df["price"].apply(lambda x: x * 2)

df["Price_Status"] = df["price"].where(
    df["price"] >= 1000,
    0
)

df["High_Price"] = df["price"].mask(
    df["price"] < 3000,
    0
)

df["price_clipped"] = df["price"].clip(
    lower=500,
    upper=5000
)

df.insert(
    2,
    "Price_Category",
    "Unknown"
)

df = df.assign(
    Price_Per_Carat=df["price"] / df["Carat"]
)

# ------------------------------------------------------------

df = df.set_index("Cut")

df = df.reset_index()

# ------------------------------------------------------------

grouped = df.groupby("Cut")

print(
    grouped["price"].agg(
        ["mean", "min", "max", "count"]
    )
)

df["Average_Cut_Price"] = (
    grouped["price"].transform("mean")
)
# ------------------------------------------------------------
cut_summary = (
    df.groupby("Cut", as_index=False)
      .agg(
          Average_Price=("price", "mean"),
          Max_Price=("price", "max"),
          Count=("price", "count")
      )
)

df = df.merge(
    cut_summary,
    on="Cut",
    how="left"
)

sample_1 = df.iloc[:100]
sample_2 = df.iloc[100:200]

combined = pd.concat(
    [sample_1, sample_2],
    ignore_index=True
)

# ------------------------------------------------------------

pivot = pd.pivot_table(
    df,
    values="price",
    index="Cut",
    columns="Color",
    aggfunc="mean"
)

melted = pd.melt(
    df,
    id_vars=["Cut", "Color"],
    value_vars=["Carat", "price"]
)

# ------------------------------------------------------------

df["Cut_List"] = df["Cut"].apply(
    lambda x: [x]
)

exploded = df.explode("Cut_List")

# ------------------------------------------------------------


print(df.memory_usage())

df.to_csv(
    "diamonds_cleaned.csv",
    index=False
)
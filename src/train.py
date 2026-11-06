"""
Train the freight rate model and write both prediction files.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

DATA = "data"
FEATURES = ["distance", "weight", "equipment", "pickup_lat", "pickup_lon",
            "delivery_lat", "delivery_lon", "market_index", "quote_signal"]
FEATURES_NO_QUOTE = [f for f in FEATURES if f != "quote_signal"]  # December file has no quote_signal


# loading and cleaning 
def load():
    train = pd.read_csv(f"{DATA}/train_test.csv")
    val = pd.read_csv(f"{DATA}/validation.csv")
    dec = pd.read_csv(f"{DATA}/december_chart_inputs.csv")
    return train, val, dec


def build_lookups(train, val):
    parts = []
    for df in [train, val]:
        for side in ["pickup", "delivery"]:
            part = df[[side, side + "_lat", side + "_lon"]]
            parts.append(part.set_axis(["city", "lat", "lon"], axis=1))
    city_coords = pd.concat(parts).drop_duplicates("city").set_index("city")

    # market index is basically one value per day
    daily_index = pd.concat([train, val]).groupby("date")["market_index"].mean()

    weight_median = train.loc[train["weight"] > 0].groupby("equipment")["weight"].median()
    return city_coords, daily_index, weight_median


def clean(df, daily_index, weight_median):
    df = df.copy()
    df["weight"] = df["weight"].abs()  # negative weights look like sign errors
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_median))
    df["market_index"] = df["market_index"].fillna(df["date"].map(daily_index))
    return df


def drop_bad_prices(train):
    # remove loads priced over 2x or under 0.5x their route's usual rate per mile
    rpm = train["posted_rate"] / train["distance"]
    ratio = rpm / rpm.groupby([train["pickup"], train["delivery"]]).transform("median")
    keep = (ratio <= 2) & (ratio >= 0.5)
    print(f"Removed {(~keep).sum()} rows with unrealistic prices")
    return train[keep].reset_index(drop=True)


def prepare_december(dec, city_coords, daily_index):
    dec = dec.copy()
    for side in ["pickup", "delivery"]:
        dec[side + "_lat"] = dec[side].map(city_coords["lat"])
        dec[side + "_lon"] = dec[side].map(city_coords["lon"])
    dec["market_index"] = dec["date"].map(daily_index)
    return dec


# Model
def make_features(df, features):
    X = df[features].copy()
    X["equipment"] = X["equipment"].map({"Dry Van": 0, "Reefer": 1, "Flatbed": 2})
    return X


def train_model(df, features):
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                          categorical_features=[features.index("equipment")],
                                          random_state=42)
    # predict rate per mile, then multiply back by distance
    model.fit(make_features(df, features), df["posted_rate"] / df["distance"])
    return model


def predict(model, df, features):
    return model.predict(make_features(df, features)) * df["distance"].values


def evaluate(train):
    # same time-based split as the notebook: train Jan-Aug, test Sep-Oct
    dates = pd.to_datetime(train["date"])
    fit, test = train[dates < "2025-09-01"], train[dates >= "2025-09-01"]
    for name, features in [("with quote_signal", FEATURES), ("without quote_signal", FEATURES_NO_QUOTE)]:
        pred = predict(train_model(fit, features), test, features)
        mae = mean_absolute_error(test["posted_rate"], pred)
        mape = mean_absolute_percentage_error(test["posted_rate"], pred) * 100
        print(f"Holdout (Sep-Oct), {name}: MAE ${mae:.0f}, MAPE {mape:.2f}%")


# Main
def main():
    train, val, dec = load()
    city_coords, daily_index, weight_median = build_lookups(train, val)

    train = drop_bad_prices(clean(train, daily_index, weight_median))
    val = clean(val, daily_index, weight_median)
    dec_ready = prepare_december(dec, city_coords, daily_index)

    evaluate(train)

    # final models use all cleaned training data
    model = train_model(train, FEATURES)
    dec_model = train_model(train, FEATURES_NO_QUOTE)

    # validation predictions, in the template's order
    template = pd.read_csv(f"{DATA}/validation_predictions_template.csv")
    val_pred = pd.Series(predict(model, val, FEATURES).round(2), index=val["load_id"])
    template["predicted_rate"] = template["load_id"].map(val_pred)
    template.to_csv("validation_predictions.csv", index=False)
    print(f"Saved validation_predictions.csv ({len(template)} rows)")

    # December: fill predicted_rate, keep the original columns
    dec["predicted_rate"] = predict(dec_model, dec_ready, FEATURES_NO_QUOTE).round(2)
    dec.to_csv(f"{DATA}/december_chart_inputs.csv", index=False)
    dec.to_csv("december_predictions.csv", index=False)
    print(f"Saved December predictions: ${dec['predicted_rate'].min():.2f} to ${dec['predicted_rate'].max():.2f}")


if __name__ == "__main__":
    main()
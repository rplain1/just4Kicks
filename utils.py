import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import polars as pl
from arviz.labels import BaseLabeller


def prep_data() -> pl.DataFrame:
    """
    Create the base dataframe combining the two data sources

    Returns:
        pl.DataFrame: dataframe of field goal attempts with kicker info
    """
    df_field_goals = pl.read_csv("data/field_goal_attempts.csv")

    df_kickers = pl.read_csv("data/kickers.csv")

    df = (
        df_field_goals.join(df_kickers, on="player_id")
        .rename({"attempt_yards": "distance"})
        .with_columns(
            success=pl.when(pl.col("field_goal_result") == "Made")
            .then(pl.lit(1))
            .otherwise(pl.lit(0)),
            player_id=pl.col("player_id").cast(pl.Utf8).cast(pl.Categorical),
            player_name=pl.col("player_name").cast(pl.Categorical),
            season_type=pl.col("season_type").cast(pl.Categorical),
        )
    )

    return df


def create_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a pandas dataframe and creates 1 row per kick attempt
    for each kicker in 2018

    Bambi relies on pandas...

    Returns:
        pd.DataFrame
    """
    df_2018 = df[df["season"] == 2018]

    # grid for kicks between 25 and 60 yards
    df_grid_18 = (
        df_2018[["player_id", "player_name"]]
        .drop_duplicates()
        .assign(key=1)
        .merge(
            pd.DataFrame({"distance": [x for x in range(25, 61)]}).assign(key=1), on="key"
        )
        .assign(__obs__=lambda x: np.arange(x.shape[0]))
    )

    return df_grid_18


def create_fgoe_draws(
    df: pd.DataFrame, model: bmb.Model, idata: az.InferenceData
) -> pd.DataFrame:
    """
    Create a dataframe where each draw contains the player predicted values
    minus the avg predicted values without groups. This is the foundation of
    any plot or dataframe for the simulated FGOE

    Args:
        df: (pd.DataFrame) pandas dataframe for the model to predict on (2018 grid)
        model: (bmb.Model) bambi model
        idata: (az.InferenceData) inference data object outputted from the model (for prediction)

    Returns:
        pl.DataFrame: chain and draw sample of player and fgoe metric
    """

    # predictions on grid without group effects, giving the average player
    avg_preds = model.predict(idata, data=df, inplace=False, include_group_specific=False)

    # posterior predictions on grid with player effects
    preds = model.predict(idata, data=df, inplace=False)

    df_pred = preds.posterior["p"].to_dataframe().reset_index()
    df_avg_pred = avg_preds["posterior"]["p"].to_dataframe().reset_index()

    # take the predictions form the players and subtract the posterior predictions without group effects.
    # This provides the main source of the metric for simulated FGOE
    df_fgoe = (
        df_pred.merge(df_avg_pred, on=["chain", "draw", "__obs__"], how="inner")
        .merge(
            df[["player_id", "player_name", "distance", "__obs__"]],
            on=["__obs__"],
            how="left",
        )
        .assign(fgoe=lambda x: x["p_x"] - x["p_y"])
    )

    # aggregation of predicted probability over expeceted
    df_output = (
        pl.from_pandas(df_fgoe)
        .group_by(["player_id", "player_name", "draw", "chain"])
        .agg(pl.col("fgoe").sum())
    )

    return df_output


def build_model(df, s="bs(distance, df=4, intercept=False)") -> bmb.Model:
    """
    Contruct the Bambi model object

    Returns:
        bmb.Model: model object from predefined formula
    """
    priors = {
        s: bmb.Prior("StudentT", nu=3, mu=0, sigma=0.5),
        "1|player_id": bmb.Prior(
            "StudentT", nu=3, mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.5)
        ),
    }

    model = bmb.Model(
        f"success ~ (1|player_id) + {s}",
        data=df,
        family="bernoulli",
        priors=priors,
    )

    return model


def create_leaderboard(df: pl.DataFrame, idata: az.InferenceData) -> pl.DataFrame:
    """
    Create the leaderboard of FGOE values

    Args:
        df: (pl.DataFrame) dataframe from `create_fgoe_draws()`
        idata: (az.InferenceData) from model

    Returns:
        pl.DataFrame: Polars dataframe of leaderboard

    """

    # perc player was in top 5 of a draw
    df_top_5 = (
        df.with_columns(
            player_name=pl.col("player_name").cast(pl.Utf8),
            rank=pl.col("fgoe").rank("dense", descending=True).over(["draw", "chain"]),
        )
        .group_by(["player_name"])
        .agg(top_5=(pl.col("rank") <= 5).mean())
        .sort("top_5", descending=True)
    )

    # summary table of the metric, with rank added and joined with top 5
    leaderboard = (
        (
            pl.from_pandas(
                az.summary(idata, var_names=["fgoe"], hdi_prob=0.9)
                .reset_index()
                .assign(player_name=lambda x: x["index"].str.extract(r"fgoe\[(.+)\]"))
            )
            .sort(["mean"], descending=True)
            .join(df_top_5, on="player_name")
            .with_row_index("rank", offset=1)
            .rename({"mean": "rating"})
            .join(
                df[["player_name", "player_id"]]
                .unique()
                .with_columns(pl.col("player_name").cast(pl.Utf8)),
                on="player_name",
            )
        )
        .select(
            [
                "player_id",
                "player_name",
                "rank",
                "rating",
                "sd",
                "hdi_5%",
                "hdi_95%",
                "top_5",
            ]
        )
        .sort(["rank"])
    )

    return leaderboard


def main():
    df = prep_data()
    df_pd = df.to_pandas()

    model = build_model(df_pd)
    idata = model.fit(draws=1000, chains=4, cores=1)

    df_grid_2018 = create_grid(df_pd)
    df_output = create_fgoe_draws(df_grid_2018, model, idata)

    idata_post = create_xarray(df_output)
    leaderboard = create_leaderboard(df_output, idata_post)

    return leaderboard


# Plotting utils ----------


def create_xarray(df: pl.DataFrame):
    """
    Convert polars dataframe with specific dimensions into Xarray
    to work with Arviz plots.

    Args:
        df: (pl.DataFrame) Dataframe from create_draws_df()
    """
    return df.to_pandas().set_index(["chain", "draw", "player_name"]).to_xarray()


# This is from an LLM, I haven't cleaned up arviz plots before
class PlayerOnlyLabeller(BaseLabeller):
    def make_label_flat(self, var_name, coord_values=(), dims=(), index=None):
        # coord_values is a dict or list of tuples — try to extract the player name
        if isinstance(coord_values, dict):
            return coord_values.get("player_name", var_name)
        elif isinstance(coord_values, (list, tuple)) and coord_values:
            return coord_values[0][1]  # (dim_name, value)
        else:
            return var_name


def plot_fgoe(idata_fgoe, title, **kwargs):
    fgoe_mean = idata_fgoe["fgoe"].mean(("chain", "draw"))

    axs = az.plot_forest(
        idata_fgoe.sortby(fgoe_mean * -1),
        var_names=["fgoe"],
        combined=True,
        labeller=PlayerOnlyLabeller(),
        **kwargs,
    )
    ax = axs[0]
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("FGOE")
    ax.tick_params(axis="y", labelsize=10)

    return ax


if __name__ == "__main__":
    main()

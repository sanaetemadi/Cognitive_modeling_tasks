
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("social_memory_data.csv")

current_age = 25

weight_context = 0.6
weight_emotion = 0.4

decay_rate_per_year = 0.08
rehearsal_boost = 0.4

df["years_since"] = current_age - df["age"]

df["initial_memory_strength"] = (
    weight_context * df["context"] +
    weight_emotion * df["emotion"]
)

df["decayed_memory_strength"] = (
    df["initial_memory_strength"] *
    ((1 - decay_rate_per_year) ** df["years_since"])
)

df["final_memory_strength"] = df["decayed_memory_strength"]
rehearsed_mask = df["rehearsal"] == 1

df.loc[rehearsed_mask, "final_memory_strength"] = (
    df.loc[rehearsed_mask, "final_memory_strength"]
    + rehearsal_boost * (1 - df.loc[rehearsed_mask, "final_memory_strength"])
)


def plot_memory_strength_over_time(df):
    plt.figure()

    plt.plot(
        df["age"],
        df["final_memory_strength"],
        marker="o"
    )

    plt.xlabel("Age at event")
    plt.ylabel("Final memory strength at age 25")
    plt.title("Retention of social episodic memories over time")
    plt.xticks(df["age"])
    plt.show()


def plot_rehearsal_effect(df):
    plt.figure()

    df_no_rehearsal = df[df["rehearsal"] == 0]
    plt.scatter(
        df_no_rehearsal["age"],
        df_no_rehearsal["final_memory_strength"],
        label="No rehearsal"
    )

    df_rehearsed = df[df["rehearsal"] == 1]
    plt.scatter(
        df_rehearsed["age"],
        df_rehearsed["final_memory_strength"],
        label="Rehearsed"
    )

    plt.xlabel("Age at event")
    plt.ylabel("Final memory strength at age 25")
    plt.title("Effect of rehearsal on social episodic memory retention")

    plt.xticks(df["age"])

    plt.legend()
    plt.show()


def plot_context_dependent_retrieval(df, similarity_high=0.8, similarity_low=0.3):

    plt.figure()

    recall_high = df["final_memory_strength"] * similarity_high
    recall_low = df["final_memory_strength"] * similarity_low

    plt.plot(
        df["age"],
        recall_high,
        marker="o",
        label="High context similarity"
    )

    plt.plot(
        df["age"],
        recall_low,
        marker="o",
        label="Low context similarity"
    )

    plt.xlabel("Age at event")
    plt.ylabel("Retrieval indicator")
    plt.title("Context-dependent retrieval of social episodic memories")

    plt.xticks(df["age"])
    plt.legend()
    plt.show()


plot_memory_strength_over_time(df)
plot_rehearsal_effect(df)
plot_context_dependent_retrieval(df)

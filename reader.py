import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

global df
df = pd.read_csv("student_habits_performance.csv")


def calcStressLevel():
    np.random.seed(8)
    noise = np.random.normal(loc=0, scale=8, size=len(df))
    mh_norm = (df["mental_health_rating"] - 1) / (10 - 1)
    df["stress_level"] = (1 - mh_norm) * 100 + noise
    df["stress_level"] = df["stress_level"].clip(0, 100)


calcStressLevel()


def question1():
    colors = df["gender"].map({
        "Female": "red",
        "Male": "black"
    }).fillna("gray")

    plt.figure(figsize=(8, 6))
    plt.scatter(df["stress_level"], df["exam_score"], c=colors, alpha=0.7)
    plt.legend(
        handles=[
            plt.Line2D([], [], marker='o', color='red',
                       linestyle='', label='Female'),
            plt.Line2D([], [], marker='o', color='black',
                       linestyle='', label='Male'),
            plt.Line2D([], [], marker='o', color='gray',
                       linestyle='', label='Other')
        ],
        title="Gender"
    )

    plt.xlabel("Stress Level")
    plt.ylabel("Exam Score")
    plt.title("Scatter Plot of Stress Level vs Exam Score")
    plt.grid(True)
    plt.show()


def question2():
    edu_to_base = {
        "None": 4.5,
        "High School": 6.0,
        "Bachelor": 7.5,
        "Master": 9.0,
    }

    df["family_support_base"] = df["parental_education_level"].map(edu_to_base)

    df["mental_health_rating"] = pd.to_numeric(
        df["mental_health_rating"], errors="coerce")

    mh_min = df["mental_health_rating"].min()
    mh_max = df["mental_health_rating"].max()
    df["mh_norm"] = (df["mental_health_rating"] - mh_min) / \
        (mh_max - mh_min + 1e-9)

    noise = np.random.normal(loc=0, scale=1.0, size=len(df))

    df["family_support_score"] = df["family_support_base"] + \
        (df["mh_norm"] * 1.2) + noise
    df["family_support_score"] = df["family_support_score"].clip(1, 10)
    df["family_support_level"] = pd.qcut(
        df["family_support_score"],
        q=3,
        labels=["Low", "Medium", "High"]
    )
    summary = df.groupby("family_support_level")[
        "exam_score"].mean().reindex(["Low", "Medium", "High"])
    cmap = plt.cm.viridis
    colors = cmap([0.2, 0.5, 0.8])
    plt.figure(figsize=(8, 6))
    plt.bar(
        summary.index.astype(str),
        summary.values,
        color=colors,
        edgecolor="black",
        linewidth=1
    )
    plt.xlabel("Family Support Level")
    plt.ylabel("Average Exam Score")
    plt.title("Average Exam Score by Family Support Level")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def question3():
    def sleep_category(hours):
        if hours < 6:
            return "Low Sleep (<6h)"
        elif 6 <= hours <= 8:
            return "Normal Sleep (6–8h)"
        else:
            return "High Sleep (>8h)"

    df["sleep_group"] = df["sleep_hours"].apply(sleep_category)

    order = ["Low Sleep (<6h)", "Normal Sleep (6–8h)", "High Sleep (>8h)"]
    df["sleep_group"] = pd.Categorical(
        df["sleep_group"], categories=order, ordered=True)

    sleep_palette = {
        "Low Sleep (<6h)": "salmon",
        "Normal Sleep (6–8h)": "skyblue",
        "High Sleep (>8h)": "lightgreen",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(
        df["sleep_hours"],
        bins=15,
        edgecolor="black",
        color="lightsteelblue",
        alpha=0.85
    )

    mean_sleep = df["sleep_hours"].mean()
    median_sleep = df["sleep_hours"].median()
    axes[0].axvline(mean_sleep, linestyle="--", linewidth=2,
                    label=f"Mean = {mean_sleep:.2f}")
    axes[0].axvline(median_sleep, linestyle=":", linewidth=2,
                    label=f"Median = {median_sleep:.2f}")

    axes[0].set_xlabel("Sleep Hours")
    axes[0].set_ylabel("Number of Students")
    axes[0].set_title("Histogram of Sleep Hours", fontsize=13)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()

    sns.boxplot(
        data=df,
        x="sleep_group",
        y="exam_score",
        order=order,
        palette=sleep_palette,
        ax=axes[1]
    )
    sns.stripplot(
        data=df,
        x="sleep_group",
        y="exam_score",
        order=order,
        color="black",
        jitter=0.2,
        alpha=0.35,
        size=3,
        ax=axes[1]
    )

    axes[1].set_xlabel("Sleep Group")
    axes[1].set_ylabel("Exam Score")
    axes[1].set_title("Exam Score by Sleep Group", fontsize=13)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Sleep Pattern and Academic Performance", fontsize=14)
    plt.tight_layout()
    plt.show()


def question4():
    df["social_media_hours_rounded"] = df["social_media_hours"].round()

    line_data = (
        df
        .groupby("social_media_hours_rounded")["stress_level"]
        .mean()
        .reset_index()
        .sort_values("social_media_hours_rounded")
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        line_data["social_media_hours_rounded"],
        line_data["stress_level"],
        marker="o"
    )

    plt.xlabel("Social Media Hours (Rounded)")
    plt.ylabel("Average Stress Level")
    plt.title("Average Stress Level vs Social Media Usage")
    plt.grid(True)
    plt.show()


def question5():
    base_df = df[["part_time_job", "exam_score", "stress_level"]].copy()

    violin_data = pd.melt(
        base_df,
        id_vars="part_time_job",
        value_vars=["exam_score", "stress_level"],
        var_name="metric",
        value_name="value"
    )

    palette = {"exam_score": "green", "stress_level": "orange"}
    order = ["No", "Yes"]
    hue_order = ["exam_score", "stress_level"]

    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=violin_data,
        x="part_time_job",
        y="value",
        hue="metric",
        split=True,
        inner="quartile",
        palette=palette,
        cut=0,
        order=order,
        hue_order=hue_order,
        scale="width"
    )

    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.xlabel("Part-Time Job (No / Yes)")
    plt.ylabel("Value (0–100 scale; Exam vs Stress)")
    plt.title("Split Violin: Exam Score vs Stress Level by Part-Time Job")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=violin_data,
        x="part_time_job",
        y="value",
        hue="metric",
        split=True,
        inner="quartile",
        palette={"exam_score": "green", "stress_level": "orange"},
        cut=0,
        order=order,
        hue_order=hue_order,
        scale="width"
    )

    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.xlabel("Part-Time Job (No / Yes)")
    plt.ylabel("Value (0–100 scale; Exam vs Stress)")
    plt.title("Split Violin: Exam Score vs Stress Level by Part-Time Job")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def question6():
    high_stress_df = df[df["stress_level"] > 70].copy()

    def teacher_quality_binary(att):
        if att < 70:
            return "Low Teacher Quality"
        else:
            return "High Teacher Quality"

    high_stress_df["Teacher_Quality"] = high_stress_df["attendance_percentage"].apply(
        teacher_quality_binary
    )

    high_stress_df["School_Resources"] = high_stress_df["internet_quality"]

    summary = (
        high_stress_df
        .groupby(["Teacher_Quality", "School_Resources"])
        .agg(
            avg_score=("exam_score", "mean"),
            count=("exam_score", "size")
        )
        .reset_index()
    )

    x_order = ["Low Teacher Quality", "High Teacher Quality"]
    hue_order = ["Poor", "Average", "Good"]

    palette = {
        "Poor": "orange",
        "Average": "blue",
        "Good": "green"
    }

    plt.figure(figsize=(9, 6))

    ax = sns.barplot(
        data=summary,
        x="Teacher_Quality",
        y="avg_score",
        hue="School_Resources",
        order=x_order,
        hue_order=hue_order,
        palette=palette
    )

    ax.legend(
        title="School_Resources",
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=1
    )

    plt.subplots_adjust(top=0.85)

    for bar, (_, row) in zip(ax.patches, summary.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xlabel("Teacher Quality")
    plt.ylabel("Average Exam Score")
    plt.title(
        "Effect of Teacher Quality and School Resources\n"
        "on High-Stress Students",
        pad=30
    )
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def question7():

    def exercise_category(x):
        if x in [0, 1, 2]:
            return "Low"
        elif x in [3, 4]:
            return "Medium"
        else:
            return "High"

    df["physical_activity_level"] = df["exercise_frequency"].apply(
        exercise_category)

    sns.boxplot(
        data=df,
        x="physical_activity_level",
        y="stress_level",
        order=["Low", "Medium", "High"],
        boxprops={"alpha": 0.5}
    )

    sns.stripplot(
        data=df,
        x="physical_activity_level",
        y="stress_level",
        order=["Low", "Medium", "High"],
        jitter=0.2
    )

    plt.title("Relationship between Physical Activity and Stress Level")
    plt.xlabel("Physical Activity Level")
    plt.ylabel("Stress Level")
    plt.show()


def question8():
    df["total_media_hours"] = df["social_media_hours"] + df["netflix_hours"]
    mean_total_media = df["total_media_hours"].mean()
    std_total_media = df["total_media_hours"].std()

    def classify_peer_influence(x):
        if x < mean_total_media - std_total_media:
            return "Positive"
        elif mean_total_media - std_total_media <= x <= mean_total_media + std_total_media:
            return "Neutral"
        else:
            return "Negative"
    df["peer_influence"] = df["total_media_hours"].apply(
        classify_peer_influence)

    sns.kdeplot(
        data=df,
        x="exam_score",
        hue="peer_influence",
        hue_order=["Negative", "Neutral", "Positive"],
        fill=True,
        alpha=0.3,
        common_norm=False

    )

    plt.title("Distribution of Exam Scores by Peer Influence")
    plt.xlabel("Exam Score")
    plt.ylabel("Density")
    plt.show()


def question9():

    gender_palette = {
        "Male": "blue",
        "Female": "red"
    }

    gender_grid = sns.FacetGrid(
        data=df,
        col="gender",
        col_order=["Male", "Female"]
    )
    gender_grid.map_dataframe(
        sns.scatterplot,
        x="study_hours_per_day",
        y="exam_score",
        hue="gender",
        palette=gender_palette,
        alpha=0.7,
        s=40
    )
    gender_grid.fig.suptitle(
        "Relationship Between Study Hours and Exam Score by Gender",
        fontsize=14
    )
    gender_grid.set_axis_labels(
        "Study Hours per Day",
        "Exam Score"
    )
    plt.subplots_adjust(top=0.85)
    plt.show()


def question10():

    sns.scatterplot(
        data=df,
        x="study_hours_per_day",
        y="exam_score",
        size="attendance_percentage",
        sizes=(20, 300),
        alpha=0.6,
        hue="attendance_percentage",
        palette="viridis"

    )

    plt.title("Study Hours vs Exam Score with Attendance as Bubble Size")
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Exam Score")
    plt.show()


def question11():
    counts = df["extracurricular_participation"].value_counts()
    labels = counts.index
    sizes = counts.values
    colors = ["lightgreen", "orange"]

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90
    )

    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    plt.gca().add_artist(centre_circle)
    plt.title("Participation in Extracurricular Activities")
    plt.tight_layout()
    plt.show()


question7()

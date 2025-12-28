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

    plt.figure(figsize=(8, 6))
    plt.bar(summary.index.astype(str), summary.values)

    plt.xlabel("Family Support Level")
    plt.ylabel("Average Exam Score")
    plt.title("Average Exam Score by Family Support Level")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def question3():
    plt.figure(figsize=(8, 6))
    plt.hist(df["sleep_hours"], bins=15, edgecolor="black")

    plt.xlabel("Sleep Hours")
    plt.ylabel("Number of Students")
    plt.title("Histogram of Sleep Hours")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    def sleep_category(hours):
        if hours < 6:
            return "Low Sleep (<6h)"
        elif 6 <= hours <= 8:
            return "Normal Sleep (6–8h)"
        else:
            return "High Sleep (>8h)"

    df["sleep_group"] = df["sleep_hours"].apply(sleep_category)

    df.boxplot(
        column="exam_score",
        by="sleep_group",
        grid=False
    )

    plt.xlabel("Sleep Group")
    plt.ylabel("Exam Score")
    plt.title("Exam Score Distribution by Sleep Group")
    plt.suptitle("")
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
    exam_score_not_working = df.loc[
        df["part_time_job"] == "No",
        "exam_score"
    ]

    exam_score_working = df.loc[
        df["part_time_job"] == "Yes",
        "exam_score"
    ]

    stress_not_working = df.loc[
        df["part_time_job"] == "No",
        "stress_level"
    ]

    stress_working = df.loc[
        df["part_time_job"] == "Yes",
        "stress_level"
    ]

    base_df = df[["part_time_job", "exam_score", "stress_level"]].copy()

    # تبدیل به فرمت long (درست و امن)
    violin_data = pd.melt(
        base_df,
        id_vars="part_time_job",
        value_vars=["exam_score", "stress_level"],
        var_name="metric",
        value_name="value"
    )

    # رنگ‌ها
    palette = {
        "exam_score": "green",
        "stress_level": "orange"
    }

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

    for i, bar in enumerate(ax.patches):
        row = summary.iloc[i]
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
        "on High-Stress Students"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def question7(): pass
def question8(): pass
def question9(): pass
def question10(): pass
def question11(): pass


question1()
question2()
question3()
question4()
question5()

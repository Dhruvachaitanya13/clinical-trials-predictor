"""
Day 5: standalone analytics dashboard (the CVS-flavored companion project).
Reuses data/trials_features.csv from Day 1.
Run: streamlit run dashboard/app.py
Add 2-3 sentences of FINDINGS under each chart -- that's the analyst skill.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/trials_features.csv")
df["completed"] = (df["overall_status"].str.upper() == "COMPLETED").astype(int)

st.title("US Clinical Trials Landscape")
st.caption("Source: AACT / ClinicalTrials.gov")

# Q1: completion rate by phase
st.subheader("Completion rate by phase")
st.plotly_chart(px.bar(
    df.groupby("phase")["completed"].mean().reset_index(),
    x="phase", y="completed"
))
st.write("Finding: [write 2-3 sentences here].")

# Q2: by sponsor class
st.subheader("Completion rate by sponsor type")
st.plotly_chart(px.bar(
    df.groupby("sponsor_class")["completed"].mean().reset_index(),
    x="sponsor_class", y="completed"
))
st.write("Finding: [write 2-3 sentences here].")

# Q3: enrollment vs completion
st.subheader("Enrollment size vs completion")
df["enroll_bucket"] = pd.qcut(df["enrollment"].clip(upper=5000), 5, duplicates="drop")
st.plotly_chart(px.bar(
    df.groupby("enroll_bucket")["completed"].mean().reset_index().astype(str),
    x="enroll_bucket", y="completed"
))
st.write("Finding: [write 2-3 sentences here].")

# Q4: trend over time
st.subheader("Completion rate over time")
trend = (df[df["start_year"].between(2008, 2022)]
         .groupby("start_year")["completed"].mean().reset_index())
st.plotly_chart(px.line(trend, x="start_year", y="completed"))
st.write("Finding: [write 2-3 sentences here].")

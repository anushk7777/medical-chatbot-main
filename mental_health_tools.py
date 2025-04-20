import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random


class MentalHealthAssessment:
    def __init__(self):
        pass

    def create_depression_course_chart(self, pattern_type):
        """Create a visualization of depression course patterns"""
        # Generate time points (weeks)
        weeks = np.arange(0, 52, 1)

        if pattern_type == "Single Episode with Recovery":
            # Create a single episode with recovery
            baseline = np.ones(52) * 10
            episode = 70 * np.exp(-0.1 * (weeks - 10) ** 2)
            depression = baseline + episode
            depression = np.clip(depression, 0, 100)

            # Create dataframe
            df = pd.DataFrame({
                'Week': weeks,
                'Severity': depression
            })

            # Create line chart
            fig = px.line(df, x='Week', y='Severity')

            # Add a vertical line for treatment start
            fig.add_vline(x=15, line_dash="dash", line_color="green",
                          annotation_text="Treatment Started", annotation_position="top right")

            # Add horizontal regions
            fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal")
            fig.add_hrect(y0=30, y1=60, line_width=0, fillcolor="orange", opacity=0.1, annotation_text="Mild-Moderate")
            fig.add_hrect(y0=60, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Severe")

        elif pattern_type == "Recurrent Depression":
            # Create recurrent episodes
            baseline = np.ones(52) * 20
            episode1 = 60 * np.exp(-0.2 * (weeks - 10) ** 2)
            episode2 = 70 * np.exp(-0.2 * (weeks - 30) ** 2)
            depression = baseline + episode1 + episode2
            depression = np.clip(depression, 0, 100)

            # Create dataframe
            df = pd.DataFrame({
                'Week': weeks,
                'Severity': depression
            })

            # Create line chart
            fig = px.line(df, x='Week', y='Severity')

            # Add vertical lines for episodes
            fig.add_vline(x=10, line_dash="dash", line_color="red",
                          annotation_text="Episode 1", annotation_position="top right")
            fig.add_vline(x=30, line_dash="dash", line_color="red",
                          annotation_text="Episode 2", annotation_position="top right")

            # Add horizontal regions
            fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal")
            fig.add_hrect(y0=30, y1=60, line_width=0, fillcolor="orange", opacity=0.1, annotation_text="Mild-Moderate")
            fig.add_hrect(y0=60, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Severe")

        elif pattern_type == "Chronic Depression":
            # Create chronic depression
            baseline = np.ones(52) * 60
            fluctuation = np.sin(weeks / 4) * 15
            depression = baseline + fluctuation
            depression = np.clip(depression, 0, 100)

            # Create dataframe
            df = pd.DataFrame({
                'Week': weeks,
                'Severity': depression
            })

            # Create line chart
            fig = px.line(df, x='Week', y='Severity')

            # Add horizontal regions
            fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal")
            fig.add_hrect(y0=30, y1=60, line_width=0, fillcolor="orange", opacity=0.1, annotation_text="Mild-Moderate")
            fig.add_hrect(y0=60, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Severe")

        elif pattern_type == "Treatment Response":
            # Create treatment response comparison
            weeks = np.arange(0, 12, 0.5)  # 12 weeks with half-week intervals

            # Create multiple treatment trajectories
            baseline = np.ones(len(weeks)) * 80  # Start with severe depression

            # Different treatment responses
            treatment1 = baseline - (weeks * 6)  # Medication
            treatment2 = baseline - (weeks * 5)  # Psychotherapy
            treatment3 = baseline - (weeks * 7)  # Combined
            placebo = baseline - (weeks * 2)  # Placebo/No treatment

            # Clip to reasonable ranges
            treatment1 = np.clip(treatment1, 20, 100)
            treatment2 = np.clip(treatment2, 25, 100)
            treatment3 = np.clip(treatment3, 15, 100)
            placebo = np.clip(placebo, 60, 100)

            # Create dataframe
            df = pd.DataFrame({
                'Week': weeks,
                'Medication': treatment1,
                'Psychotherapy': treatment2,
                'Combined Treatment': treatment3,
                'No Treatment': placebo
            })

            # Melt dataframe for plotly
            df_melted = pd.melt(df, id_vars=['Week'],
                                value_vars=['Medication', 'Psychotherapy', 'Combined Treatment', 'No Treatment'],
                                var_name='Treatment', value_name='Severity')

            # Create line chart
            fig = px.line(df_melted, x='Week', y='Severity', color='Treatment',
                          color_discrete_map={
                              'Medication': '#1f77b4',
                              'Psychotherapy': '#ff7f0e',
                              'Combined Treatment': '#2ca02c',
                              'No Treatment': '#d62728'
                          })

            # Add horizontal regions
            fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Normal")
            fig.add_hrect(y0=30, y1=60, line_width=0, fillcolor="orange", opacity=0.1, annotation_text="Mild-Moderate")
            fig.add_hrect(y0=60, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Severe")

        # Update layout
        fig.update_layout(
            title=f"Depression Course: {pattern_type}",
            xaxis_title="Weeks",
            yaxis_title="Symptom Severity",
            yaxis=dict(range=[0, 100]),
            height=400,
            margin=dict(l=20, r=20, t=50, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def get_pattern_description(self, pattern_type):
        """Get description of depression pattern"""
        descriptions = {
            "Single Episode with Recovery": """
            **Single Episode with Recovery** is the most favorable course of depression. 

            Key characteristics:
            - One discrete episode lasting typically 3-9 months
            - Symptoms gradually improve with treatment
            - Full recovery with return to baseline functioning
            - About 50-60% of people experience this pattern with their first depressive episode

            Research shows that early treatment significantly increases the likelihood of this pattern.
            """,

            "Recurrent Depression": """
            **Recurrent Depression** is characterized by multiple episodes separated by periods of recovery.

            Key characteristics:
            - Episodes may occur at random or be triggered by stressors
            - Periods between episodes are relatively symptom-free
            - Each recurrence increases risk of future episodes
            - About 80% of people with an initial depressive episode will experience recurrence

            Maintenance treatment may be recommended for those with multiple recurrences.
            """,

            "Chronic Depression": """
            **Chronic Depression** involves persistent symptoms lasting 2+ years with minimal periods of relief.

            Key characteristics:
            - Symptoms fluctuate but never fully remit
            - Significant impact on quality of life and functioning
            - Often more resistant to standard treatments
            - Affects approximately 20-30% of people with depression

            Combined medication and psychotherapy approaches are typically most effective for chronic depression.
            """,

            "Treatment Response": """
            **Treatment Response Patterns** show how different interventions affect depression over time.

            Key observations:
            - Combined treatments (medication + therapy) typically show the best outcomes
            - Medication often works faster in the first 4-6 weeks
            - Psychotherapy may have more enduring effects after treatment ends
            - Without treatment, depression symptoms tend to persist or worsen

            Response to treatment is usually seen within 4-8 weeks, but full recovery may take longer.
            """
        }

        return descriptions.get(pattern_type, "No description available.")

    def create_treatment_comparison_chart(self, depression_type):
        """Create a visualization comparing treatment effectiveness"""
        # Set up data based on depression type
        if depression_type == "Major Depression":
            treatments = [
                "SSRIs", "SNRIs", "Bupropion", "TCAs", "CBT",
                "Psychodynamic", "Combined Med+CBT", "TMS", "ECT"
            ]
            response_rates = [65, 63, 60, 55, 58, 55, 78, 50, 80]
            remission_rates = [40, 38, 35, 30, 32, 30, 45, 30, 55]

        elif depression_type == "Treatment-Resistant Depression":
            treatments = [
                "Medication Switch", "Medication Augmentation", "CBT+Medication",
                "TMS", "ECT", "Ketamine", "Esketamine", "VNS"
            ]
            response_rates = [25, 35, 40, 45, 70, 60, 57, 40]
            remission_rates = [15, 20, 25, 30, 50, 40, 37, 25]

        elif depression_type == "Depression with Anxiety":
            treatments = [
                "SSRIs", "SNRIs", "Benzodiazepines", "CBT",
                "CBT for Anxiety", "Combined Med+CBT", "Mindfulness"
            ]
            response_rates = [70, 72, 60, 60, 65, 80, 55]
            remission_rates = [45, 47, 30, 40, 42, 55, 35]

        elif depression_type == "Bipolar Depression":
            treatments = [
                "Lithium", "Lamotrigine", "Quetiapine", "Lurasidone",
                "Olanzapine+Fluoxetine", "Antidepressant Monotherapy", "Psychotherapy"
            ]
            response_rates = [40, 50, 60, 55, 65, 35, 45]
            remission_rates = [25, 30, 40, 35, 45, 20, 30]

        # Create dataframe
        df = pd.DataFrame({
            'Treatment': treatments,
            'Response Rate (%)': response_rates,
            'Remission Rate (%)': remission_rates
        })

        # Sort by response rate
        df = df.sort_values('Response Rate (%)', ascending=False)

        # Create grouped bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df['Treatment'],
            y=df['Response Rate (%)'],
            name='Response Rate',
            marker_color='#1f77b4'
        ))

        fig.add_trace(go.Bar(
            x=df['Treatment'],
            y=df['Remission Rate (%)'],
            name='Remission Rate',
            marker_color='#2ca02c'
        ))

        # Update layout
        fig.update_layout(
            title=f"Treatment Effectiveness for {depression_type}",
            xaxis_title="Treatment",
            yaxis_title="Rate (%)",
            yaxis=dict(range=[0, 100]),
            height=500,
            margin=dict(l=20, r=20, t=50, b=100),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode='group'
        )

        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)

        return fig

    def get_treatment_recommendations(self, depression_type):
        """Get evidence-based treatment recommendations"""
        recommendations = {
            "Major Depression": """
            **Evidence-Based Treatment Options for Major Depression:**

            **First-line treatments:**
            - SSRIs (e.g., sertraline, escitalopram) are typically first-line due to favorable side effect profile
            - Individual CBT (Cognitive Behavioral Therapy) shows comparable efficacy to medication
            - Combined medication and CBT provides the highest response rates

            **For more severe depression:**
            - SNRIs or dual-action antidepressants may be more effective
            - Combined treatment is strongly recommended
            - For severe, treatment-resistant cases, ECT shows the highest remission rates

            **Treatment course:**
            - Medication response typically begins in 2-4 weeks, with full effects at 6-8 weeks
            - CBT typically requires 12-16 weekly sessions
            - Continuation treatment (to prevent relapse) is recommended for 6-12 months after remission
            """,

            "Treatment-Resistant Depression": """
            **Evidence-Based Approaches for Treatment-Resistant Depression:**

            **Definition:** Depression that fails to respond to at least two adequate trials of different antidepressants.

            **Treatment approaches in order of typical consideration:**
            1. Optimization of current medication (ensure adequate dose and duration)
            2. Switching to a different class of antidepressant
            3. Augmentation strategies:
               - Adding an atypical antipsychotic (e.g., aripiprazole, quetiapine)
               - Adding lithium or thyroid hormone
            4. Combining antidepressants from different classes
            5. Adding evidence-based psychotherapy (particularly CBT)

            **Neuromodulation therapies:**
            - TMS (Transcranial Magnetic Stimulation) for moderate treatment resistance
            - ECT (Electroconvulsive Therapy) remains the most effective option for severe cases
            - Esketamine nasal spray shows promising results for TRD

            **Emerging treatments:**
            - Ketamine and psychedelic-assisted therapy show promising results but require specialized settings
            """,

            "Depression with Anxiety": """
            **Evidence-Based Approaches for Depression with Anxiety:**

            **Pharmacotherapy:**
            - SSRIs and SNRIs are effective for both conditions and are first-line treatments
            - SSRIs (especially escitalopram and sertraline) may have a more favorable side effect profile
            - SNRIs (especially venlafaxine and duloxetine) may be more effective for certain anxiety symptoms
            - Benzodiazepines should be limited to short-term use due to tolerance and dependence concerns

            **Psychotherapy:**
            - CBT that addresses both depression and anxiety shows the best outcomes
            - Mindfulness-based interventions show promise, particularly for preventing relapse
            - Exposure techniques may be incorporated for specific anxiety components

            **Combined approach:**
            - Combining medication and psychotherapy shows superior outcomes compared to either alone
            - This approach addresses both biological and psychological aspects of the comorbid condition
            """,

            "Bipolar Depression": """
            **Evidence-Based Approaches for Bipolar Depression:**

            **Important considerations:**
            - Treatment differs significantly from unipolar depression
            - Antidepressant monotherapy is generally not recommended due to risk of mood switches
            - Mood stabilizer foundation is essential

            **First-line treatments:**
            - Quetiapine has strong evidence as monotherapy
            - Lurasidone (with or without lithium/valproate)
            - Olanzapine-fluoxetine combination
            - Lamotrigine (particularly for prevention of future episodes)

            **Other approaches:**
            - Lithium is more effective for bipolar I than bipolar II depression
            - Valproate has moderate evidence
            - Cariprazine is FDA-approved for bipolar I depression

            **Psychotherapy adjuncts:**
            - Interpersonal and Social Rhythm Therapy (IPSRT) 
            - CBT adapted for bipolar disorder
            - Family-focused therapy

            **Caution:** Traditional antidepressants used without mood stabilizers increase risk of switching to mania/hypomania or developing rapid cycling
            """
        }

        return recommendations.get(depression_type, "No specific recommendations available.")

    def calculate_risk_score(self, gender, age_group, family_history, chronic_illness, recent_trauma, social_support):
        """Calculate depression risk score based on research-backed factors"""
        # Base risk
        risk_score = 0

        # Initialize factor weights dictionary
        factor_weights = {}

        # Gender factor (2x risk for women)
        if gender == "Female":
            risk_score += 10
            factor_weights["Female Gender"] = 10
        else:
            factor_weights["Male Gender"] = 0

        # Age factor (U-shaped curve)
        if age_group == "18-29" or age_group == "60+":
            risk_score += 10
            factor_weights[f"Age Group ({age_group})"] = 10
        elif age_group == "30-44":
            risk_score += 5
            factor_weights[f"Age Group ({age_group})"] = 5
        else:
            risk_score += 3
            factor_weights[f"Age Group ({age_group})"] = 3

        # Family history (1.5-3x risk)
        if family_history == "Yes":
            risk_score += 15
            factor_weights["Family History"] = 15
        else:
            factor_weights["Family History"] = 0

        # Chronic illness (2x risk)
        if chronic_illness == "Yes":
            risk_score += 12
            factor_weights["Chronic Illness"] = 12
        else:
            factor_weights["Chronic Illness"] = 0

        # Recent trauma/stressful event (major risk factor)
        if recent_trauma == "Yes":
            risk_score += 20
            factor_weights["Recent Stressful Event"] = 20
        else:
            factor_weights["Recent Stressful Event"] = 0

        # Social support (protective factor)
        if social_support == "Low":
            risk_score += 15
            factor_weights["Low Social Support"] = 15
        elif social_support == "Medium":
            risk_score += 5
            factor_weights["Medium Social Support"] = 5
        else:
            risk_score -= 5
            factor_weights["High Social Support"] = -5

        # Add some random variation (5%)
        variation = random.uniform(-2.5, 2.5)
        risk_score += variation

        # Ensure risk score is between 0 and 100
        risk_score = max(0, min(100, risk_score))

        return risk_score, factor_weights

    def get_risk_recommendations(self, risk_score, gender, age_group, family_history, chronic_illness, recent_trauma,
                                 social_support):
        """Get recommendations based on risk factors"""
        recommendations = [
            "Regular physical activity (30+ minutes most days) has been shown to reduce depression risk by up to 45%",
            "Maintain a consistent sleep schedule, aiming for 7-9 hours of quality sleep each night"
        ]

        # Add general recommendations
        if risk_score < 20:
            recommendations.extend([
                "Continue your healthy habits and monitor for changes in mood or energy",
                "Practice stress management techniques like deep breathing or mindfulness",
                "Stay socially connected with friends and family"
            ])
        elif risk_score < 50:
            recommendations.extend([
                "Consider learning basic cognitive-behavioral techniques to manage negative thought patterns",
                "Establish a regular routine that includes enjoyable activities",
                "Track your mood to identify patterns or triggers"
            ])
        else:
            recommendations.extend([
                "Consider consulting with a mental health professional for a screening assessment",
                "Learn about the warning signs of depression",
                "Create a wellness plan that includes self-care, social support, and professional help if needed"
            ])

        # Add factor-specific recommendations
        if family_history == "Yes":
            recommendations.append(
                "With a family history of depression, be vigilant about early signs and seek help promptly if symptoms develop")

        if chronic_illness == "Yes":
            recommendations.append(
                "For chronic physical conditions, work with your healthcare provider to optimize treatment and discuss mental health impacts")

        if recent_trauma == "Yes":
            recommendations.append(
                "After stressful life events, prioritize processing your emotions through journaling, talking with trusted others, or professional support")

        if social_support == "Low":
            recommendations.append(
                "Building social connections through community activities, support groups, or volunteering can reduce depression risk")

        # Add age-specific recommendations
        if age_group == "18-29":
            recommendations.append(
                "Young adults benefit from stress management skills and establishing healthy routines during this transitional life period")
        elif age_group == "60+":
            recommendations.append(
                "Older adults should maintain social engagement and meaningful activities, particularly after retirement or loss events")

        return recommendations

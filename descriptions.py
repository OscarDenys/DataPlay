"""
Test descriptions, guidance, and educational content for statistical testing.
Provides comprehensive information about each statistical test and testing concepts.
"""

from typing import Dict, Any

def get_test_categories() -> Dict[str, Dict[str, Any]]:
    """
    Get organized categories of statistical tests with descriptions.
    
    Returns:
        Dictionary mapping test categories to their descriptions and included tests
    """
    return {
        "Comparing Groups": {
            "description": "Tests for comparing measurements between groups",
            "when_to_use": [
                "Comparing performance between different groups",
                "Analyzing differences in outcomes across categories",
                "Evaluating treatment effects"
            ],
            "tests": [
                "Student's t-test",
                "ANOVA",
                "Mann-Whitney U Test",
                "Kruskal-Wallis H Test"
            ]
        },
        "Analyzing Relationships": {
            "description": "Tests for analyzing relationships between variables",
            "when_to_use": [
                "Understanding correlations between measurements",
                "Analyzing associations between categories",
                "Predicting values based on other variables"
            ],
            "tests": [
                "Pearson Correlation",
                "Chi-Square Test",
                "Linear Regression"
            ]
        },
        "Paired Comparisons": {
            "description": "Tests for comparing paired or matched measurements",
            "when_to_use": [
                "Before-after comparisons",
                "Matched pairs analysis",
                "Repeated measurements"
            ],
            "tests": [
                "Wilcoxon Signed-Rank Test"
            ]
        },
        "Model Evaluation": {
            "description": "Tests for evaluating model parameters",
            "when_to_use": [
                "Testing regression coefficients",
                "Evaluating model parameters",
                "Comparing against theoretical values"
            ],
            "tests": [
                "Wald Test"
            ]
        }
    }

def get_test_selection_guide() -> Dict[str, Dict[str, str]]:
    """
    Get guidance for test selection based on data characteristics.
    
    Returns:
        Dictionary mapping data scenarios to appropriate tests
    """
    return {
        "Two Groups": {
            "Numeric, Normal": "Student's t-test",
            "Numeric, Non-normal": "Mann-Whitney U Test",
            "Categorical": "Chi-Square Test",
            "help": """
            For comparing two groups:
            1. Check if your data is numeric or categorical
            2. For numeric data, check normality using Q-Q plots
            3. Choose test based on data type and normality
            """
        },
        "Three+ Groups": {
            "Numeric, Normal": "ANOVA",
            "Numeric, Non-normal": "Kruskal-Wallis H Test",
            "help": """
            For comparing three or more groups:
            1. Verify independent groups
            2. Check normality for each group
            3. Consider post-hoc tests if main test is significant
            """
        },
        "Paired Data": {
            "Numeric, Normal": "Paired t-test",
            "Numeric, Non-normal": "Wilcoxon Signed-Rank Test",
            "help": """
            For paired/matched data:
            1. Confirm data is truly paired
            2. Check normality of differences
            3. Consider temporal effects
            """
        },
        "Relationships": {
            "Numeric vs Numeric": "Pearson Correlation",
            "Categorical vs Categorical": "Chi-Square Test",
            "Predicting Values": "Linear Regression",
            "help": """
            For analyzing relationships:
            1. Identify variable types
            2. Check for linear relationships
            3. Consider direction of relationship
            """
        }
    }

def get_assumption_guide() -> Dict[str, Dict[str, Any]]:
    """
    Get guidance on testing and interpreting statistical assumptions.
    
    Returns:
        Dictionary mapping assumptions to their descriptions and checks
    """
    return {
        "Normality": {
            "description": "Data follows a normal distribution",
            "importance": """
            Normality is important because many statistical tests assume the data follows
            a normal distribution. Violations can lead to unreliable results.
            """,
            "how_to_check": [
                "Visual inspection with Q-Q plots",
                "Shapiro-Wilk test for smaller samples",
                "Anderson-Darling test for larger samples"
            ],
            "what_if_violated": """
            If normality is violated:
            1. Consider transforming the data
            2. Use non-parametric alternatives
            3. For large samples, rely on Central Limit Theorem
            """
        },
        "Homogeneity of Variance": {
            "description": "Groups have similar variances",
            "importance": """
            Equal variances ensure that the spread of data is similar across groups,
            which is important for comparing means accurately.
            """,
            "how_to_check": [
                "Levene's test",
                "Visual inspection with box plots",
                "Comparison of standard deviations"
            ],
            "what_if_violated": """
            If variances are unequal:
            1. Use Welch's t-test instead of Student's t-test
            2. Consider transforming the data
            3. Use non-parametric alternatives
            """
        },
        "Independence": {
            "description": "Observations are independent of each other",
            "importance": """
            Independence ensures that each observation provides unique information
            and is not influenced by other observations.
            """,
            "how_to_check": [
                "Study design review",
                "Time series plots for temporal data",
                "Durbin-Watson test for regression"
            ],
            "what_if_violated": """
            If independence is violated:
            1. Use tests for dependent samples
            2. Consider hierarchical/multilevel models
            3. Account for clustering in the analysis
            """
        }
    }

def get_test_description(test_type: str) -> Dict[str, Any]:
    """
    Get comprehensive description and guidance for each test type.
    
    Args:
        test_type: Name of the statistical test
        
    Returns:
        Dictionary containing test information and guidance
    """
    descriptions = {
        "Student's t-test": {
            "purpose": "Compare means between two independent groups of continuous data",
            "null_hypothesis": "There is no significant difference between the means of the two groups",
            "alternative_hypothesis": "There is a significant difference between the means of the two groups",
            "assumptions": [
                "Data in each group is normally distributed",
                "Homogeneity of variances between groups",
                "Independent observations",
                "Continuous dependent variable",
                "Adequate sample size (generally n > 30 per group)"
            ],
            "when_to_use": [
                "Comparing average sales between two regions",
                "Analyzing difference in test scores between two teaching methods",
                "Evaluating treatment effects in A/B testing",
                "Comparing performance metrics between two groups"
            ],
            "example_hypothesis": {
                "context": "Comparing customer satisfaction scores between two store locations",
                "null": "H₀: μ₁ = μ₂ (mean satisfaction scores are equal)",
                "alternative": "H₁: μ₁ ≠ μ₂ (mean satisfaction scores are different)"
            },
            "common_pitfalls": [
                "Using with small samples without checking normality",
                "Applying to non-independent samples (use paired t-test instead)",
                "Ignoring outliers that might affect normality",
                "Not checking for homogeneity of variances"
            ],
            "required_data": [
                "One categorical variable with exactly two groups",
                "One continuous dependent variable"
            ],
            "interpretation_guide": {
                "p_value": "If p < α, reject null hypothesis; significant difference exists",
                "effect_size": "Cohen's d: 0.2 (small), 0.5 (medium), 0.8 (large)"
            }
        },
        
        "ANOVA": {
            "purpose": "Compare means across three or more independent groups",
            "null_hypothesis": "All group means are equal",
            "alternative_hypothesis": "At least one group mean is different from the others",
            "assumptions": [
                "Normal distribution within each group",
                "Homogeneity of variances across groups",
                "Independent observations",
                "Continuous dependent variable",
                "No significant outliers"
            ],
            "when_to_use": [
                "Comparing performance across multiple departments",
                "Analyzing effects of different treatment levels",
                "Evaluating differences across multiple customer segments",
                "Testing impact of multiple marketing strategies"
            ],
            "example_hypothesis": {
                "context": "Comparing average sales across four different regions",
                "null": "H₀: μ₁ = μ₂ = μ₃ = μ₄ (all regional means are equal)",
                "alternative": "H₁: At least one regional mean differs"
            },
            "common_pitfalls": [
                "Not conducting post-hoc tests when ANOVA is significant",
                "Using with small group sizes",
                "Ignoring the assumption of homogeneity of variances",
                "Not checking for normality within groups"
            ],
            "required_data": [
                "One categorical variable with three or more groups",
                "One continuous dependent variable"
            ],
            "interpretation_guide": {
                "p_value": "If p < α, reject null hypothesis; differences exist between groups",
                "effect_size": "η² (Eta squared): 0.01 (small), 0.06 (medium), 0.14 (large)"
            }
        },

        "Chi-Square Test": {
            "purpose": "Test for association between categorical variables or goodness-of-fit",
            "null_hypothesis": "No association exists between the categorical variables",
            "alternative_hypothesis": "An association exists between the categorical variables",
            "assumptions": [
                "Independent observations",
                "Mutually exclusive categories",
                "Expected frequencies ≥ 5 in each cell",
                "Random sampling",
                "Adequate sample size"
            ],
            "when_to_use": [
                "Testing association between gender and product preference",
                "Analyzing relationship between education level and career choice",
                "Examining independence of demographic factors",
                "Testing if observed frequencies match expected proportions"
            ],
            "example_hypothesis": {
                "context": "Testing if product preference is associated with gender",
                "null": "H₀: No association between gender and product preference",
                "alternative": "H₁: Association exists between gender and product preference"
            },
            "common_pitfalls": [
                "Using with small expected frequencies",
                "Including dependent samples",
                "Using with ordinal or continuous data",
                "Not considering Fisher's exact test for 2x2 tables with small samples"
            ],
            "required_data": [
                "Two or more categorical variables",
                "Frequency counts or raw categorical data"
            ],
            "interpretation_guide": {
                "p_value": "If p < α, reject null hypothesis; association exists",
                "effect_size": "Cramer's V: 0.1 (small), 0.3 (medium), 0.5 (large)"
            }
        },

        "Mann-Whitney U Test": {
            "purpose": "Compare distributions between two independent groups (non-parametric alternative to t-test)",
            "null_hypothesis": "The distributions of both groups are equal",
            "alternative_hypothesis": "The distributions of the groups are different",
            "assumptions": [
                "Independent observations",
                "Similar shaped distributions (if comparing medians)",
                "Ordinal or continuous dependent variable",
                "Random sampling"
            ],
            "when_to_use": [
                "Data not normally distributed",
                "Small sample sizes",
                "Ordinal data (e.g., Likert scales)",
                "Comparing groups with outliers"
            ],
            "example_hypothesis": {
                "context": "Comparing customer satisfaction ratings between two service types",
                "null": "H₀: Distributions are equal",
                "alternative": "H₁: Distributions differ"
            },
            "common_pitfalls": [
                "Using with paired samples (use Wilcoxon signed-rank instead)",
                "Interpreting as purely median comparison",
                "Using with nominal data",
                "Not checking distribution shapes when comparing medians"
            ],
            "required_data": [
                "One categorical variable with two groups",
                "One ordinal or continuous dependent variable"
            ],
            "interpretation_guide": {
                "p_value": "If p < α, reject null hypothesis; distributions differ",
                "effect_size": "r = Z/√N: 0.1 (small), 0.3 (medium), 0.5 (large)"
            }
        }
    }
    
    return descriptions.get(test_type, {})

def get_effect_size_guide() -> Dict[str, Dict[str, Any]]:
    """
    Get guidance on interpreting effect sizes for different tests.
    
    Returns:
        Dictionary mapping effect size measures to their interpretations
    """
    return {
        "Cohen's d": {
            "description": "Standardized difference between two means",
            "thresholds": {
                "small": 0.2,
                "medium": 0.5,
                "large": 0.8
            },
            "interpretation": """
            - Small (0.2): Subtle effect, might be meaningful in large samples
            - Medium (0.5): Moderate effect, noticeable to the naked eye
            - Large (0.8): Substantial effect, clearly visible
            """,
            "examples": [
                "Height difference between men and women (d ≈ 2.0)",
                "Effect of aspirin on pain (d ≈ 0.4)",
                "Effect of caffeine on alertness (d ≈ 0.3)"
            ]
        },
        "Correlation (r)": {
            "description": "Strength and direction of linear relationship",
            "thresholds": {
                "small": 0.1,
                "medium": 0.3,
                "large": 0.5
            },
            "interpretation": """
            - Small (0.1): Weak relationship, might need large sample to detect
            - Medium (0.3): Moderate relationship, typically visible in scatterplot
            - Large (0.5): Strong relationship, clear pattern visible
            """,
            "examples": [
                "Height and weight (r ≈ 0.8)",
                "Education and income (r ≈ 0.4)",
                "Age and reaction time (r ≈ -0.3)"
            ]
        },
        "Eta-squared (η²)": {
            "description": "Proportion of variance explained in ANOVA",
            "thresholds": {
                "small": 0.01,
                "medium": 0.06,
                "large": 0.14
            },
            "interpretation": """
            - Small (0.01): 1% of variance explained
            - Medium (0.06): 6% of variance explained
            - Large (0.14): 14% of variance explained
            """,
            "examples": [
                "Effect of teaching method on test scores (η² ≈ 0.05)",
                "Effect of diet on weight loss (η² ≈ 0.12)",
                "Effect of drug dose on response (η² ≈ 0.20)"
            ]
        }
    }
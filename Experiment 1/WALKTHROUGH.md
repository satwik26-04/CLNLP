# Walkthrough: Experiment 1 - Employee Data Analysis

This walkthrough provides a comprehensive, line-by-line explanation of all code, functions, and syntax implemented in [Experiment_1.ipynb](file:///c:/Users/navee/Documents/CLNLP%20Lab/Experiment%201/Experiment_1.ipynb), along with data analysis interpretations for all 10 sub-experiments.

---

## 1. Environment Setup & Dataset Loading

### Code Executed
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv('employee_information_100.csv')
df.head()
```

### Line-by-Line & Syntax Explanation
1. `import pandas as pd`
   - **Syntax**: `import <library> as <alias>`
   - **Explanation**: Imports the **pandas** library and binds it to the short alias `pd`. Pandas is used for tabular data manipulation, DataFrame structures, aggregations, and filtering.
2. `import matplotlib.pyplot as plt`
   - **Syntax**: `import <module> as <alias>`
   - **Explanation**: Imports the `pyplot` module from the **matplotlib** visualization framework under the alias `plt` to control figure creation, layout, axes, labels, and rendering.
3. `import seaborn as sns`
   - **Syntax**: `import <library> as <alias>`
   - **Explanation**: Imports **seaborn**, a high-level statistical plotting library built on top of Matplotlib that provides attractive default aesthetics and color palettes.
4. `sns.set_theme(style="whitegrid")`
   - **Syntax**: `sns.set_theme(style: str)`
   - **Explanation**: Configures the global aesthetic theme for all subsequent plots. `"whitegrid"` adds subtle grid lines against a light background for enhanced readability.
5. `df = pd.read_csv('employee_information_100.csv')`
   - **Syntax**: `pandas.read_csv(filepath_or_buffer: str) -> DataFrame`
   - **Explanation**: Reads the CSV dataset from disk and converts it into a 2-dimensional labeled data structure known as a pandas `DataFrame`, assigned to variable `df`.
6. `df.head()`
   - **Syntax**: `DataFrame.head(n: int = 5) -> DataFrame`
   - **Explanation**: Returns the first 5 records of the DataFrame to inspect columns (`Employee_ID`, `Name`, `Department`, `Age`, `Gender`, `Salary`, `Experience_Years`) and data types.

---

## Experiment 1.1: Average Salary by Department (Bar Chart)

### Objective
Calculate the mean salary across each department and visualize the results using a bar chart.

### Code Executed
```python
avg_salary = df.groupby('Department')['Salary'].mean().reset_index()
print(avg_salary)

plt.figure(figsize=(8, 5))
sns.barplot(data=avg_salary, x='Department', y='Salary', hue='Department', palette='viridis', legend=False)
plt.title('Average Salary by Department')
plt.xlabel('Department')
plt.ylabel('Average Salary ($)')
plt.show()
```

### Line-by-Line & Syntax Explanation
1. `avg_salary = df.groupby('Department')['Salary'].mean().reset_index()`
   - `df.groupby('Department')`: Groups DataFrame rows by unique values in the `'Department'` column.
   - `['Salary']`: Selects the `'Salary'` column for aggregation.
   - `.mean()`: Computes the arithmetic average of salaries in each department group.
   - `.reset_index()`: Converts the grouped Series index back into a standard column, producing a neat 2-column DataFrame (`Department`, `Salary`).
2. `print(avg_salary)`
   - Prints the tabular numerical values of each department's average salary.
3. `plt.figure(figsize=(8, 5))`
   - Initializes a new Matplotlib figure window with a width of 8 inches and a height of 5 inches.
4. `sns.barplot(data=avg_salary, x='Department', y='Salary', hue='Department', palette='viridis', legend=False)`
   - `data=avg_salary`: Specifies the source DataFrame.
   - `x='Department'`: Assigns department names to the horizontal axis.
   - `y='Salary'`: Assigns the calculated average salary to the vertical bar height.
   - `hue='Department'`: Assigns distinct color encoding per department.
   - `palette='viridis'`: Uses the high-contrast `'viridis'` perceptually uniform color gradient.
   - `legend=False`: Suppresses duplicate legend labels since x-axis ticks already indicate departments.
5. `plt.title('Average Salary by Department')`
   - Adds a descriptive title above the plot.
6. `plt.xlabel('Department')` & `plt.ylabel('Average Salary ($)')`
   - Labels the X and Y axes with meaningful dimension names and units.
7. `plt.show()`
   - Renders and displays the active plot figure.

### Output Summary
- **Operations**: $90,875.00 *(Highest average compensation)*
- **Finance**: $82,416.67
- **Marketing**: $76,100.00
- **Sales**: $74,769.23
- **IT**: $73,611.11
- **HR**: $72,095.24

---

## Experiment 1.2: Number of Employees in Each Department

### Objective
Determine the employee headcount per department and plot the distribution.

### Code Executed
```python
dept_counts = df['Department'].value_counts().reset_index()
dept_counts.columns = ['Department', 'Employee_Count']
print(dept_counts)

plt.figure(figsize=(8, 5))
sns.barplot(data=dept_counts, x='Department', y='Employee_Count', hue='Department', palette='mako', legend=False)
plt.title('Employee Count by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.show()
```

### Line-by-Line & Syntax Explanation
1. `dept_counts = df['Department'].value_counts().reset_index()`
   - `df['Department'].value_counts()`: Counts the frequency of occurrence for each unique department in descending order.
   - `.reset_index()`: Flattens the resulting Series into a two-column DataFrame.
2. `dept_counts.columns = ['Department', 'Employee_Count']`
   - Explicitly renames DataFrame columns to `'Department'` and `'Employee_Count'` for clarity.
3. `print(dept_counts)`
   - Outputs the frequency table.
4. `plt.figure(figsize=(8, 5))`
   - Sets the plotting canvas dimensions (8x5 inches).
5. `sns.barplot(data=dept_counts, x='Department', y='Employee_Count', hue='Department', palette='mako', legend=False)`
   - Draws vertical bars representing department size using the sleek `'mako'` color palette.
6. `plt.title('Employee Count by Department')`, `plt.xlabel(...)`, `plt.ylabel(...)`
   - Sets chart title and axes labels.
7. `plt.show()`
   - Renders the visualization.

### Output Summary
- **HR**: 21 employees *(Largest team)*
- **Marketing**: 20 employees
- **IT**: 18 employees
- **Operations**: 16 employees
- **Sales**: 13 employees
- **Finance**: 12 employees *(Smallest team)*

---

## Experiment 1.3: Percentage of Male and Female Employees (Pie Chart)

### Objective
Compute the gender breakdown across all 100 employees and render a proportional pie chart.

### Code Executed
```python
gender_counts = df['Gender'].value_counts()
print(gender_counts)

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4C72B0', '#DD8452'], explode=(0.05, 0))
plt.title('Gender Distribution')
plt.show()
```

### Line-by-Line & Syntax Explanation
1. `gender_counts = df['Gender'].value_counts()`
   - Calculates the total count of `'Male'` and `'Female'` entries.
2. `print(gender_counts)`
   - Displays the raw gender counts (`Male: 64`, `Female: 36`).
3. `plt.figure(figsize=(6, 6))`
   - Creates a 1:1 square aspect ratio canvas to keep the pie chart circular.
4. `plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4C72B0', '#DD8452'], explode=(0.05, 0))`
   - `gender_counts`: Numerical slice sizes.
   - `labels=gender_counts.index`: Slices labeled as `'Male'` and `'Female'`.
   - `autopct='%1.1f%%'`: Formats slice percentage text to 1 decimal place.
   - `startangle=90`: Rotates the first slice starting angle to 90 degrees (12 o'clock).
   - `colors=['#4C72B0', '#DD8452']`: Applies refined blue and coral slice colors.
   - `explode=(0.05, 0)`: Slightly offsets the first slice outward for visual emphasis.
5. `plt.title('Gender Distribution')`
   - Adds the chart header.
6. `plt.show()`
   - Displays the pie chart.

### Output Summary
- **Male**: 64 (64.0%)
- **Female**: 36 (36.0%)

---

## Experiment 1.4: Salary Distribution (Histogram)

### Objective
Visualize the spread, skewness, and frequency of employee compensation across salary bins.

### Code Executed
```python
plt.figure(figsize=(8, 5))
sns.histplot(df['Salary'], bins=10, kde=True, color='teal')
plt.title('Salary Distribution')
plt.xlabel('Salary ($)')
plt.ylabel('Frequency')
plt.show()
```

### Line-by-Line & Syntax Explanation
1. `plt.figure(figsize=(8, 5))`
   - Allocates an 8x5 inch plot figure.
2. `sns.histplot(df['Salary'], bins=10, kde=True, color='teal')`
   - `df['Salary']`: 1D numerical data series to partition into bins.
   - `bins=10`: Divides the salary span into 10 equal intervals.
   - `kde=True`: Enables **Kernel Density Estimation** to draw a smooth probability density curve over the histogram bars.
   - `color='teal'`: Applies a modern teal fill color.
3. `plt.title('Salary Distribution')`, `plt.xlabel('Salary ($)')`, `plt.ylabel('Frequency')`
   - Configures title and axis annotations.
4. `plt.show()`
   - Renders the salary distribution graph.

### Output Summary
- Salaries range from **$30,000** to **$119,000**.
- The median salary is **$78,000** with a mean of **$77,760**.
- The distribution is relatively balanced across the mid-to-high tiers with peaks around $60k, $80k, and $115k.

---

## Experiment 1.5: Relationship Between Experience and Salary (Scatter Plot)

### Objective
Investigate whether higher years of experience correspond to higher salaries across different departments.

### Code Executed
```python
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Experience_Years', y='Salary', hue='Department', s=70)
plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

### Line-by-Line & Syntax Explanation
1. `plt.figure(figsize=(8, 5))`
   - Initializes the plot canvas.
2. `sns.scatterplot(data=df, x='Experience_Years', y='Salary', hue='Department', s=70)`
   - `data=df`: Input dataset.
   - `x='Experience_Years'`: Places work tenure (0 to 35 years) on the X axis.
   - `y='Salary'`: Places salary compensation on the Y axis.
   - `hue='Department'`: Colors points according to the employee's department.
   - `s=70`: Increases individual marker size to 70 points for visual clarity.
3. `plt.title('Experience vs Salary')`, `plt.xlabel(...)`, `plt.ylabel(...)`
   - Standard chart title and axes labeling.
4. `plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')`
   - Positions the legend outside the plotting area on the top-right to prevent obscuring data points.
5. `plt.show()`
   - Renders the scatter plot.

### Output Summary
- Correlation between experience and salary is `r ≈ 0.033`, indicating salary is distributed across senior and junior roles without strict linear dependency on years of experience alone.

---

## Experiment 1.6: Top 10 Employees with Highest Salary

### Objective
Extract and present the 10 highest-earning employees in the organization.

### Code Executed
```python
top_10_salary = df.nlargest(10, 'Salary')
top_10_salary[['Employee_ID', 'Name', 'Department', 'Salary', 'Experience_Years']]
```

### Line-by-Line & Syntax Explanation
1. `top_10_salary = df.nlargest(10, 'Salary')`
   - `DataFrame.nlargest(n: int, columns: str)`: Efficiently retrieves the `n=10` rows with the largest values in the `'Salary'` column, sorted in descending order.
2. `top_10_salary[['Employee_ID', 'Name', 'Department', 'Salary', 'Experience_Years']]`
   - Filters the DataFrame to show only key attributes in the output table.

### Output Summary
| Employee_ID | Name | Department | Salary ($) | Experience (Yrs) |
| :--- | :--- | :--- | :--- | :--- |
| **E086** | Omar86 | Marketing | 119,000 | 25 |
| **E088** | Diana88 | Marketing | 119,000 | 23 |
| **E015** | Ishan15 | Operations | 117,000 | 20 |
| **E023** | Rahul23 | Finance | 117,000 | 7 |
| **E059** | Neha59 | IT | 117,000 | 12 |
| **E049** | Charlie49 | IT | 116,000 | 33 |
| **E032** | George32 | Operations | 115,000 | 26 |
| **E056** | Sneha56 | Operations | 115,000 | 8 |
| **E065** | Bob65 | Finance | 115,000 | 27 |
| **E010** | Vikram10 | IT | 114,000 | 14 |

---

## Experiment 1.7: Highest Salary in Every Department

### Objective
Determine the peak compensation figure achievable within each department.

### Code Executed
```python
highest_salary_dept = df.groupby('Department')['Salary'].max().reset_index()
highest_salary_dept
```

### Line-by-Line & Syntax Explanation
1. `highest_salary_dept = df.groupby('Department')['Salary'].max().reset_index()`
   - `df.groupby('Department')`: Partitions the DataFrame by department.
   - `['Salary']`: Targets the compensation column.
   - `.max()`: Computes the maximum value per department group.
   - `.reset_index()`: Resets the grouping index to standard tabular columns.
2. `highest_salary_dept`
   - Evaluates and displays the resulting table in the notebook output.

### Output Summary
- **Marketing**: $119,000
- **Finance**: $117,000
- **IT**: $117,000
- **Operations**: $117,000
- **Sales**: $114,000
- **HR**: $113,000

---

## Experiment 1.8: Employees with Salary Greater than Overall Average Salary

### Objective
Filter for all personnel earning above the global company mean compensation ($77,760.00).

### Code Executed
```python
overall_avg_salary = df['Salary'].mean()
print(f"Overall Average Salary: {overall_avg_salary:.2f}")

above_avg_employees = df[df['Salary'] > overall_avg_salary]
print(f"Total employees with above-average salary: {len(above_avg_employees)}")
above_avg_employees[['Employee_ID', 'Name', 'Department', 'Salary', 'Experience_Years']].head(10)
```

### Line-by-Line & Syntax Explanation
1. `overall_avg_salary = df['Salary'].mean()`
   - Computes the global mean of the `'Salary'` column across all 100 records.
2. `print(f"Overall Average Salary: {overall_avg_salary:.2f}")`
   - Formats the float value to 2 decimal places using Python f-strings.
3. `above_avg_employees = df[df['Salary'] > overall_avg_salary]`
   - **Boolean Indexing**: Evaluates the condition `df['Salary'] > overall_avg_salary` per row (returning `True`/`False` mask) and selects only rows where the expression evaluates to `True`.
4. `print(f"Total employees with above-average salary: {len(above_avg_employees)}")`
   - Prints the count of employees satisfying the criterion (51 out of 100).
5. `above_avg_employees[['Employee_ID', 'Name', 'Department', 'Salary', 'Experience_Years']].head(10)`
   - Displays the first 10 matching records.

---

## Experiment 1.9: Average Years of Experience in Each Department

### Objective
Evaluate departmental seniority and average employee tenure.

### Code Executed
```python
avg_exp_dept = df.groupby('Department')['Experience_Years'].mean().reset_index()
avg_exp_dept
```

### Line-by-Line & Syntax Explanation
1. `avg_exp_dept = df.groupby('Department')['Experience_Years'].mean().reset_index()`
   - `df.groupby('Department')`: Groups records by department name.
   - `['Experience_Years']`: Targets the experience metric.
   - `.mean()`: Calculates the mean experience in years for each department.
   - `.reset_index()`: Creates a structured 2-column DataFrame (`Department`, `Experience_Years`).
2. `avg_exp_dept`
   - Displays the computed averages.

### Output Summary
- **Sales**: 23.85 years *(Most experienced department on average)*
- **Marketing**: 21.90 years
- **HR**: 19.81 years
- **IT**: 18.39 years
- **Operations**: 16.88 years
- **Finance**: 11.17 years

---

## Experiment 1.10: Age Distribution of Employees (Histogram)

### Objective
Analyze workforce demographic spread across age cohorts.

### Code Executed
```python
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=10, kde=True, color='coral')
plt.title('Employee Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

### Line-by-Line & Syntax Explanation
1. `plt.figure(figsize=(8, 5))`
   - Allocates the Matplotlib figure canvas.
2. `sns.histplot(df['Age'], bins=10, kde=True, color='coral')`
   - `df['Age']`: Targets employee ages (range: 22 to 59 years).
   - `bins=10`: Bins ages into 10 intervals spanning ~3.7 years each.
   - `kde=True`: Draws the Kernel Density Estimation curve showing age trends.
   - `color='coral'`: Applies an orange-coral tone.
3. `plt.title('Employee Age Distribution')`, `plt.xlabel('Age')`, `plt.ylabel('Frequency')`
   - Sets chart title and axis labels.
4. `plt.show()`
   - Displays the final histogram.

### Output Summary
- Employee ages range from **22 to 59 years**.
- Mean age: **41.93 years**, Median age: **42.0 years**, Standard deviation: **11.60 years**.
- The workforce demonstrates a healthy demographic spread across junior (20s), mid-career (30s-40s), and senior (50s) cohorts.

---

## Notebook Compliance Verification
- **Code comments**: `0` (Strictly zero comments in all code cells as requested).
- **Markdown cells**: Concise, clean headers and objectives preceding each cell.
- **Execution state**: All 23 cells fully executed with inline outputs, tables, and rendered charts.

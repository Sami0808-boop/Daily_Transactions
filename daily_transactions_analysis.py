#!/usr/bin/env python3
"""
Daily Transactions Analysis Project
Analyzing household financial transactions to identify spending patterns,
budget optimization opportunities, and financial insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Additional libraries for advanced analysis
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)

# Machine Learning Libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class DailyTransactionsAnalyzer:
    def __init__(self, file_path='Daily Household Transactions.csv'):
        """
        Initialize the Daily Transactions Analyzer
        
        Parameters:
        file_path (str): Path to the CSV file containing transaction data
        """
        self.file_path = file_path
        self.df = None
        self.df_clean = None
        self.insights = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("📊 Loading Daily Transactions Dataset...")
        
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📈 Shape: {self.df.shape}")
            print(f"📅 Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            
            # Display basic info
            print("\n📋 Dataset Information:")
            print(self.df.info())
            
            # Display first few rows
            print("\n🔍 First 5 rows:")
            print(self.df.head())
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def clean_data(self):
        """Clean and preprocess the transaction data"""
        print("\n🧹 Cleaning and Preprocessing Data...")
        
        # Create a copy for cleaning
        self.df_clean = self.df.copy()
        
        # Check for missing values
        print("\n📊 Missing Values Analysis:")
        missing_values = self.df_clean.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Handle missing values
        self.df_clean['Subcategory'].fillna('Uncategorized', inplace=True)
        self.df_clean['Note'].fillna('No description', inplace=True)
        
        # Convert Date column to datetime
        self.df_clean['Date'] = pd.to_datetime(self.df_clean['Date'], format='%d/%m/%y %H:%M:%S')
        
        # Extract additional date features
        self.df_clean['Year'] = self.df_clean['Date'].dt.year
        self.df_clean['Month'] = self.df_clean['Date'].dt.month
        self.df_clean['Day'] = self.df_clean['Date'].dt.day
        self.df_clean['DayOfWeek'] = self.df_clean['Date'].dt.day_name()
        self.df_clean['Hour'] = self.df_clean['Date'].dt.hour
        self.df_clean['Quarter'] = self.df_clean['Date'].dt.quarter
        
        # Create time-based categories
        self.df_clean['TimeOfDay'] = pd.cut(
            self.df_clean['Hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
        
        # Separate income and expenses
        self.df_clean['Is_Income'] = (self.df_clean['Income/Expense'] == 'Income').astype(int)
        self.df_clean['Is_Expense'] = (self.df_clean['Income/Expense'] == 'Expense').astype(int)
        
        # Create amount categories for expenses
        self.df_clean['Amount_Category'] = pd.cut(
            self.df_clean['Amount'],
            bins=[0, 100, 500, 1000, 5000, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        print("✅ Data cleaning completed!")
        print(f"📊 Clean dataset shape: {self.df_clean.shape}")
        
        return True
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("\n🔍 Performing Exploratory Data Analysis...")
        
        # Basic statistics
        print("\n📈 Basic Statistics:")
        print(self.df_clean.describe())
        
        # Transaction type distribution
        print(f"\n💰 Transaction Type Distribution:")
        transaction_dist = self.df_clean['Income/Expense'].value_counts()
        print(transaction_dist)
        
        # Total income and expenses
        total_income = self.df_clean[self.df_clean['Income/Expense'] == 'Income']['Amount'].sum()
        total_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense']['Amount'].sum()
        net_savings = total_income - total_expenses
        
        self.insights['total_income'] = total_income
        self.insights['total_expenses'] = total_expenses
        self.insights['net_savings'] = net_savings
        self.insights['savings_rate'] = (net_savings / total_income) * 100 if total_income > 0 else 0
        
        print(f"\n💵 Financial Summary:")
        print(f"   Total Income: ₹{total_income:,.2f}")
        print(f"   Total Expenses: ₹{total_expenses:,.2f}")
        print(f"   Net Savings: ₹{net_savings:,.2f}")
        print(f"   Savings Rate: {self.insights['savings_rate']:.2f}%")
        
        # Create comprehensive visualizations
        self.create_visualizations()
        
        return True
    
    def create_visualizations(self):
        """Create comprehensive visualizations for the analysis"""
        print("\n📊 Creating Visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Transaction Type Distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Daily Transactions Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Transaction type pie chart
        transaction_counts = self.df_clean['Income/Expense'].value_counts()
        axes[0, 0].pie(transaction_counts.values, labels=transaction_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Transaction Type Distribution')
        
        # Payment mode distribution
        mode_counts = self.df_clean['Mode'].value_counts().head(5)
        axes[0, 1].bar(range(len(mode_counts)), mode_counts.values)
        axes[0, 1].set_xticks(range(len(mode_counts)))
        axes[0, 1].set_xticklabels(mode_counts.index, rotation=45)
        axes[0, 1].set_title('Top 5 Payment Modes')
        axes[0, 1].set_ylabel('Number of Transactions')
        
        # Top categories
        top_categories = self.df_clean['Category'].value_counts().head(8)
        axes[0, 2].barh(range(len(top_categories)), top_categories.values)
        axes[0, 2].set_yticks(range(len(top_categories)))
        axes[0, 2].set_yticklabels(top_categories.index)
        axes[0, 2].set_title('Top 8 Transaction Categories')
        axes[0, 2].set_xlabel('Number of Transactions')
        
        # Amount distribution
        axes[1, 0].hist(self.df_clean['Amount'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Transaction Amount Distribution')
        axes[1, 0].set_xlabel('Amount (₹)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Monthly spending trend
        monthly_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('Month')['Amount'].sum()
        axes[1, 1].plot(monthly_expenses.index, monthly_expenses.values, marker='o', linewidth=2)
        axes[1, 1].set_title('Monthly Expenses Trend')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Total Expenses (₹)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Day of week spending
        dow_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('DayOfWeek')['Amount'].sum()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_expenses = dow_expenses.reindex(dow_order)
        axes[1, 2].bar(range(len(dow_expenses)), dow_expenses.values)
        axes[1, 2].set_xticks(range(len(dow_expenses)))
        axes[1, 2].set_xticklabels(dow_expenses.index, rotation=45)
        axes[1, 2].set_title('Spending by Day of Week')
        axes[1, 2].set_ylabel('Total Expenses (₹)')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Advanced Analysis Plots
        self.create_advanced_plots()
        
        return True
    
    def create_advanced_plots(self):
        """Create advanced analysis plots"""
        
        # Create subplots for advanced analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Transaction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Category-wise spending analysis
        category_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
        top_categories = category_expenses.head(10)
        
        axes[0, 0].barh(range(len(top_categories)), top_categories.values)
        axes[0, 0].set_yticks(range(len(top_categories)))
        axes[0, 0].set_yticklabels(top_categories.index)
        axes[0, 0].set_title('Top 10 Categories by Total Spending')
        axes[0, 0].set_xlabel('Total Amount (₹)')
        
        # 2. Payment mode vs amount
        mode_amount = self.df_clean.groupby('Mode')['Amount'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        top_modes = mode_amount.head(8)
        
        axes[0, 1].scatter(top_modes['count'], top_modes['mean'], s=100, alpha=0.7)
        for i, mode in enumerate(top_modes.index):
            axes[0, 1].annotate(mode, (top_modes['count'].iloc[i], top_modes['mean'].iloc[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('Number of Transactions')
        axes[0, 1].set_ylabel('Average Transaction Amount (₹)')
        axes[0, 1].set_title('Payment Mode Analysis')
        
        # 3. Time of day spending pattern
        time_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('TimeOfDay')['Amount'].sum()
        axes[1, 0].pie(time_expenses.values, labels=time_expenses.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Spending by Time of Day')
        
        # 4. Monthly income vs expenses
        monthly_income = self.df_clean[self.df_clean['Income/Expense'] == 'Income'].groupby('Month')['Amount'].sum()
        monthly_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('Month')['Amount'].sum()
        
        x = range(1, 13)
        axes[1, 1].bar([i-0.2 for i in x], monthly_income.values, width=0.4, label='Income', alpha=0.7)
        axes[1, 1].bar([i+0.2 for i in x], monthly_expenses.values, width=0.4, label='Expenses', alpha=0.7)
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Amount (₹)')
        axes[1, 1].set_title('Monthly Income vs Expenses')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 3. Interactive Plotly visualizations
        self.create_interactive_plots()
        
        return True
    
    def create_interactive_plots(self):
        """Create interactive Plotly visualizations"""
        
        # 1. Time series of daily expenses
        daily_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('Date')['Amount'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_expenses['Date'],
            y=daily_expenses['Amount'],
            mode='lines+markers',
            name='Daily Expenses',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title='Daily Expenses Over Time',
            xaxis_title='Date',
            yaxis_title='Total Daily Expenses (₹)',
            hovermode='x unified',
            template='plotly_white'
        )
        fig.show()
        
        # 2. Category spending breakdown
        category_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
        
        fig = px.pie(
            values=category_expenses.values,
            names=category_expenses.index,
            title='Expense Distribution by Category'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.show()
        
        # 3. Payment mode analysis
        mode_analysis = self.df_clean.groupby('Mode').agg({
            'Amount': ['sum', 'mean', 'count']
        }).round(2)
        mode_analysis.columns = ['Total_Amount', 'Average_Amount', 'Transaction_Count']
        mode_analysis = mode_analysis.sort_values('Total_Amount', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(name='Total Amount', x=mode_analysis.index, y=mode_analysis['Total_Amount']),
            go.Bar(name='Transaction Count', x=mode_analysis.index, y=mode_analysis['Transaction_Count'] * 100)
        ])
        
        fig.update_layout(
            title='Payment Mode Analysis',
            xaxis_title='Payment Mode',
            yaxis_title='Amount (₹) / Count (x100)',
            barmode='group',
            template='plotly_white'
        )
        fig.show()
    
    def spending_pattern_analysis(self):
        """Analyze spending patterns and provide insights"""
        print("\n🔍 Analyzing Spending Patterns...")
        
        # 1. Category-wise analysis
        category_analysis = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('Category').agg({
            'Amount': ['sum', 'mean', 'count'],
            'Date': ['min', 'max']
        }).round(2)
        
        category_analysis.columns = ['Total_Spent', 'Average_Amount', 'Transaction_Count', 'First_Transaction', 'Last_Transaction']
        category_analysis = category_analysis.sort_values('Total_Spent', ascending=False)
        
        print("\n📊 Top 10 Spending Categories:")
        print(category_analysis.head(10))
        
        # 2. Monthly spending analysis
        monthly_analysis = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('Month').agg({
            'Amount': ['sum', 'mean', 'count']
        }).round(2)
        
        monthly_analysis.columns = ['Total_Spent', 'Average_Amount', 'Transaction_Count']
        
        print("\n📅 Monthly Spending Analysis:")
        print(monthly_analysis)
        
        # 3. Payment mode efficiency
        mode_efficiency = self.df_clean.groupby('Mode').agg({
            'Amount': ['sum', 'mean', 'count']
        }).round(2)
        
        mode_efficiency.columns = ['Total_Amount', 'Average_Amount', 'Transaction_Count']
        mode_efficiency = mode_efficiency.sort_values('Total_Amount', ascending=False)
        
        print("\n💳 Payment Mode Analysis:")
        print(mode_efficiency.head(8))
        
        # 4. Time-based patterns
        time_patterns = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('TimeOfDay')['Amount'].agg(['sum', 'mean', 'count']).round(2)
        time_patterns.columns = ['Total_Spent', 'Average_Amount', 'Transaction_Count']
        
        print("\n⏰ Time-based Spending Patterns:")
        print(time_patterns)
        
        # Store insights
        self.insights['top_spending_category'] = category_analysis.index[0]
        self.insights['top_spending_amount'] = category_analysis['Total_Spent'].iloc[0]
        self.insights['avg_monthly_expense'] = monthly_analysis['Total_Spent'].mean()
        self.insights['most_used_payment_mode'] = mode_efficiency.index[0]
        
        return category_analysis, monthly_analysis, mode_efficiency, time_patterns
    
    def budget_optimization_recommendations(self):
        """Generate budget optimization recommendations"""
        print("\n💡 Generating Budget Optimization Recommendations...")
        
        # Analyze spending patterns
        category_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
        
        # Calculate percentages
        total_expenses = category_expenses.sum()
        category_percentages = (category_expenses / total_expenses * 100).round(2)
        
        print("\n📊 Current Spending Breakdown:")
        for category, percentage in category_percentages.head(10).items():
            print(f"   {category}: {percentage}% (₹{category_expenses[category]:,.2f})")
        
        # Identify potential savings opportunities
        print("\n🎯 Potential Savings Opportunities:")
        
        # High-frequency, low-amount categories (daily expenses)
        daily_expenses = self.df_clean[
            (self.df_clean['Income/Expense'] == 'Expense') & 
            (self.df_clean['Category'].isin(['Food', 'Transportation', 'Household']))
        ].groupby('Category')['Amount'].agg(['sum', 'mean', 'count']).round(2)
        
        print("\n🍽️ Daily Expenses Analysis:")
        print(daily_expenses)
        
        # Recommendations
        recommendations = []
        
        # Food spending analysis
        food_expenses = self.df_clean[
            (self.df_clean['Income/Expense'] == 'Expense') & 
            (self.df_clean['Category'] == 'Food')
        ]
        
        if len(food_expenses) > 0:
            avg_food_expense = food_expenses['Amount'].mean()
            if avg_food_expense > 200:  # Assuming 200 is a reasonable daily food budget
                recommendations.append(f"Consider reducing daily food expenses. Current average: ₹{avg_food_expense:.2f}")
        
        # Transportation analysis
        transport_expenses = self.df_clean[
            (self.df_clean['Income/Expense'] == 'Expense') & 
            (self.df_clean['Category'] == 'Transportation')
        ]
        
        if len(transport_expenses) > 0:
            avg_transport_expense = transport_expenses['Amount'].mean()
            if avg_transport_expense > 100:  # Assuming 100 is a reasonable daily transport budget
                recommendations.append(f"Consider carpooling or public transport. Current average: ₹{avg_transport_expense:.2f}")
        
        # Subscription analysis
        subscription_expenses = self.df_clean[
            (self.df_clean['Income/Expense'] == 'Expense') & 
            (self.df_clean['Category'] == 'subscription')
        ]
        
        if len(subscription_expenses) > 0:
            total_subscriptions = subscription_expenses['Amount'].sum()
            recommendations.append(f"Review subscriptions. Total spent: ₹{total_subscriptions:,.2f}")
        
        # Investment analysis
        investment_income = self.df_clean[
            (self.df_clean['Income/Expense'] == 'Income') & 
            (self.df_clean['Category'].str.contains('Investment|Mutual Fund|Dividend', case=False))
        ]['Amount'].sum()
        
        if investment_income > 0:
            recommendations.append(f"Good job on investments! Income from investments: ₹{investment_income:,.2f}")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return recommendations
    
    def generate_report(self):
        """Generate a comprehensive financial report"""
        print("\n📋 Generating Comprehensive Financial Report...")
        
        # Calculate key metrics
        total_transactions = len(self.df_clean)
        total_income = self.insights['total_income']
        total_expenses = self.insights['total_expenses']
        net_savings = self.insights['net_savings']
        savings_rate = self.insights['savings_rate']
        
        # Date range
        date_range = f"{self.df_clean['Date'].min().strftime('%B %Y')} to {self.df_clean['Date'].max().strftime('%B %Y')}"
        
        print("\n" + "="*80)
        print("📊 DAILY TRANSACTIONS FINANCIAL REPORT")
        print("="*80)
        print(f"📅 Analysis Period: {date_range}")
        print(f"📈 Total Transactions: {total_transactions:,}")
        print("="*80)
        
        print("\n💰 FINANCIAL SUMMARY:")
        print(f"   💵 Total Income: ₹{total_income:,.2f}")
        print(f"   💸 Total Expenses: ₹{total_expenses:,.2f}")
        print(f"   💰 Net Savings: ₹{net_savings:,.2f}")
        print(f"   📊 Savings Rate: {savings_rate:.2f}%")
        
        print("\n🏆 TOP INSIGHTS:")
        print(f"   🥇 Highest Spending Category: {self.insights['top_spending_category']} (₹{self.insights['top_spending_amount']:,.2f})")
        print(f"   💳 Most Used Payment Mode: {self.insights['most_used_payment_mode']}")
        print(f"   📅 Average Monthly Expenses: ₹{self.insights['avg_monthly_expense']:,.2f}")
        
        # Spending patterns
        print("\n📊 SPENDING PATTERNS:")
        category_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
        print("   Top 5 Spending Categories:")
        for i, (category, amount) in enumerate(category_expenses.head(5).items(), 1):
            percentage = (amount / total_expenses * 100)
            print(f"      {i}. {category}: ₹{amount:,.2f} ({percentage:.1f}%)")
        
        # Payment mode analysis
        print("\n💳 PAYMENT MODE ANALYSIS:")
        mode_usage = self.df_clean['Mode'].value_counts().head(5)
        for i, (mode, count) in enumerate(mode_usage.items(), 1):
            percentage = (count / total_transactions * 100)
            print(f"   {i}. {mode}: {count} transactions ({percentage:.1f}%)")
        
        # Time-based insights
        print("\n⏰ TIME-BASED INSIGHTS:")
        time_expenses = self.df_clean[self.df_clean['Income/Expense'] == 'Expense'].groupby('TimeOfDay')['Amount'].sum()
        for time, amount in time_expenses.items():
            percentage = (amount / total_expenses * 100)
            print(f"   {time}: ₹{amount:,.2f} ({percentage:.1f}%)")
        
        print("\n" + "="*80)
        print("📈 RECOMMENDATIONS FOR BUDGET OPTIMIZATION")
        print("="*80)
        
        recommendations = self.budget_optimization_recommendations()
        
        print("\n" + "="*80)
        print("✅ REPORT GENERATION COMPLETE")
        print("="*80)
        
        return {
            'total_transactions': total_transactions,
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_savings': net_savings,
            'savings_rate': savings_rate,
            'date_range': date_range,
            'recommendations': recommendations
        }
    
    def run_complete_analysis(self):
        """Run the complete daily transactions analysis pipeline"""
        print("🚀 Starting Daily Transactions Analysis")
        print("="*60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Clean data
        if not self.clean_data():
            return False
        
        # Step 3: Exploratory Data Analysis
        if not self.exploratory_data_analysis():
            return False
        
        # Step 4: Spending Pattern Analysis
        self.spending_pattern_analysis()
        
        # Step 5: Generate Report
        report = self.generate_report()
        
        print("\n✅ Analysis Complete!")
        print("="*60)
        
        return report

def main():
    """Main function to run the daily transactions analysis"""
    
    # Initialize analyzer
    analyzer = DailyTransactionsAnalyzer()
    
    # Run complete analysis
    report = analyzer.run_complete_analysis()
    
    return analyzer, report

if __name__ == "__main__":
    # Run the analysis
    analyzer, report = main()
    
    # Additional insights
    print("\n" + "="*60)
    print("🔍 ADDITIONAL INSIGHTS")
    print("="*60)
    
    # Analyze transaction frequency
    daily_transaction_counts = analyzer.df_clean.groupby(analyzer.df_clean['Date'].dt.date).size()
    avg_daily_transactions = daily_transaction_counts.mean()
    max_daily_transactions = daily_transaction_counts.max()
    
    print(f"\n📊 Transaction Frequency:")
    print(f"   Average daily transactions: {avg_daily_transactions:.1f}")
    print(f"   Maximum daily transactions: {max_daily_transactions}")
    
    # Analyze income sources
    income_sources = analyzer.df_clean[analyzer.df_clean['Income/Expense'] == 'Income']['Category'].value_counts()
    print(f"\n💰 Income Sources:")
    for source, count in income_sources.items():
        amount = analyzer.df_clean[
            (analyzer.df_clean['Income/Expense'] == 'Income') & 
            (analyzer.df_clean['Category'] == source)
        ]['Amount'].sum()
        print(f"   {source}: {count} transactions, ₹{amount:,.2f}")
    
    # Analyze expense patterns by day of week
    dow_expenses = analyzer.df_clean[analyzer.df_clean['Income/Expense'] == 'Expense'].groupby('DayOfWeek')['Amount'].sum()
    print(f"\n📅 Spending by Day of Week:")
    for day, amount in dow_expenses.items():
        print(f"   {day}: ₹{amount:,.2f}") 

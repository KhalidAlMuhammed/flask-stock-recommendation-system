from flask import Flask, render_template, request, jsonify, session
import os
import google.generativeai as genai
import yfinance as yf
import math
import json
import markdown2

app = Flask(__name__)
app.secret_key = '9b05817d47f4043ab17212afe9d13343'  # Replace with a real secret key

def fetch_stock_data(ticker):
  stock = yf.Ticker(ticker)
  info = stock.info
  print(info)
  return {
    'info': info,
    'current_price': info.get('currentPrice'),
    'pe_ratio': info.get('trailingPE'),
    'forward_pe': info.get('forwardPE'),
    'price_to_book': info.get('priceToBook'),
    'dividend_yield': info.get('dividendYield', 0),
    'payout_ratio': info.get('payoutRatio', 0),
    'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
    'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
    'fifty_two_week_change': info.get('52WeekChange'),
    'analyst_target_price': info.get('targetMeanPrice'),
    'analyst_recommendation': info.get('recommendationMean'),
    'total_cash': info.get('totalCash'),
    'total_debt': info.get('totalDebt'),
    'current_ratio': info.get('currentRatio'),
    'net_income': info.get('netIncomeToCommon'),
    'roe': info.get('returnOnEquity'),
    'profit_margin': info.get('profitMargins'),
    'market_cap': info.get('marketCap'),
    'beta': info.get('beta'),
    'audit_risk': info.get('auditRisk'),
    'board_risk': info.get('boardRisk')
  }

def evaluate_stock(ticker):
  data = fetch_stock_data(ticker)
  
  evaluation = {}
  
  # Valuation
  evaluation['valuation'] = {
    'current_price': data['current_price'],
    'pe_ratio': data['pe_ratio'],
    'pe_analysis': 'High' if data['pe_ratio'] and not math.isnan(data['pe_ratio']) and data['pe_ratio'] > 25 else 'Moderate' if data['pe_ratio'] and not math.isnan(data['pe_ratio']) and data['pe_ratio'] > 15 else 'Low',
    'forward_pe': data['forward_pe'],
    'forward_pe_trend': 'Improving' if data['forward_pe'] and not math.isnan(data['forward_pe']) and data['forward_pe'] < data['pe_ratio'] else 'Worsening',
    'price_to_book': data['price_to_book'],
    'price_to_book_analysis': 'High' if data['price_to_book'] and not math.isnan(data['price_to_book']) and data['price_to_book'] > 3 else 'Moderate' if data['price_to_book'] and not math.isnan(data['price_to_book']) and data['price_to_book'] > 1 else 'Low'
  }
  
  # Dividends
  evaluation['dividends'] = {
    'dividend_yield': data['dividend_yield'],
    'yield_attractiveness': 'High' if data['dividend_yield'] and not math.isnan(data['dividend_yield']) and data['dividend_yield'] > 0.04 else 'Moderate' if data['dividend_yield'] and not math.isnan(data['dividend_yield']) and data['dividend_yield'] > 0.02 else 'Low',
    'payout_ratio': data['payout_ratio'],
    'payout_sustainability': 'Sustainable' if data['payout_ratio'] and not math.isnan(data['payout_ratio']) and data['payout_ratio'] < 0.6 else 'Potentially Unsustainable'
  }
  
  # Growth and Performance
  evaluation['growth_and_performance'] = {
    '52_week_range': f"{data['fifty_two_week_low']} - {data['fifty_two_week_high']}",
    'current_price_position': (data['current_price'] - data['fifty_two_week_low']) / (data['fifty_two_week_high'] - data['fifty_two_week_low']) if data['current_price'] and data['fifty_two_week_low'] and data['fifty_two_week_high'] else None,
    '52_week_change': data['fifty_two_week_change'],
    'performance': 'Outperforming' if data['fifty_two_week_change'] and not math.isnan(data['fifty_two_week_change']) and data['fifty_two_week_change'] > 0 else 'Underperforming'
  }
  
  # Analyst Opinions
  evaluation['analyst_opinions'] = {
    'target_price': data['analyst_target_price'],
    'upside_potential': (data['analyst_target_price'] - data['current_price']) / data['current_price'] if data['analyst_target_price'] and data['current_price'] and not math.isnan(data['analyst_target_price']) and not math.isnan(data['current_price']) else None,
    'recommendation': data['analyst_recommendation'],
    'buy_signal': 'Strong Buy' if data['analyst_recommendation'] and not math.isnan(data['analyst_recommendation']) and data['analyst_recommendation'] < 2 else 'Buy' if data['analyst_recommendation'] and not math.isnan(data['analyst_recommendation']) and data['analyst_recommendation'] < 3 else 'Hold' if data['analyst_recommendation'] and not math.isnan(data['analyst_recommendation']) and data['analyst_recommendation'] < 4 else 'Sell'
  }
  
  # Financial Health
  evaluation['financial_health'] = {
    'cash_position': data['total_cash'],
    'total_debt': data['total_debt'],
    'net_debt': data['total_debt'] - data['total_cash'] if data['total_debt'] and data['total_cash'] and not math.isnan(data['total_debt']) and not math.isnan(data['total_cash']) else None,
    'current_ratio': data['current_ratio'],
    'liquidity': 'Strong' if data['current_ratio'] and not math.isnan(data['current_ratio']) and data['current_ratio'] > 2 else 'Good' if data['current_ratio'] and not math.isnan(data['current_ratio']) and data['current_ratio'] > 1 else 'Weak'
  }
  
  # Profitability
  evaluation['profitability'] = {
    'net_income': data['net_income'],
    'roe': data['roe'],
    'roe_analysis': 'Strong' if data['roe'] and not math.isnan(data['roe']) and data['roe'] > 0.2 else 'Good' if data['roe'] and not math.isnan(data['roe']) and data['roe'] > 0.1 else 'Weak',
    'profit_margin': data['profit_margin'],
    'margin_analysis': 'High' if data['profit_margin'] and not math.isnan(data['profit_margin']) and data['profit_margin'] > 0.2 else 'Moderate' if data['profit_margin'] and not math.isnan(data['profit_margin']) and data['profit_margin'] > 0.1 else 'Low'
  }
  
  # Market Position
  evaluation['market_position'] = {
    'market_cap': data['market_cap'],
    'size_category': 'Large Cap' if data['market_cap'] and not math.isnan(data['market_cap']) and data['market_cap'] > 10e9 else 'Mid Cap' if data['market_cap'] and not math.isnan(data['market_cap']) and data['market_cap'] > 2e9 else 'Small Cap'
  }
  
  # Risks
  evaluation['risks'] = {
    'beta': data['beta'],
    'volatility': 'High' if data['beta'] and not math.isnan(data['beta']) and data['beta'] > 1.5 else 'Moderate' if data['beta'] and not math.isnan(data['beta']) and data['beta'] > 0.5 else 'Low',
    'audit_risk': data['audit_risk'],
    'board_risk': data['board_risk'],
    'governance_concern': 'High' if data['audit_risk'] and data['board_risk'] and not math.isnan(data['audit_risk']) and not math.isnan(data['board_risk']) and (data['audit_risk'] + data['board_risk']) / 2 > 7 else 'Moderate' if data['audit_risk'] and data['board_risk'] and not math.isnan(data['audit_risk']) and not math.isnan(data['board_risk']) and (data['audit_risk'] + data['board_risk']) / 2 > 5 else 'Low'
  }
  evaluation['full_info'] = data['info']
    
  return evaluation

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        stock_evaluation = evaluate_stock(ticker)
        
        # Configure Google AI
        genai.configure(api_key="AIzaSyDSYuKyOG8nQkNK4PNJqtIyrKJkfIeBgUQ")

        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        "# Instructions for Stock Analysis\n\nAs an LLM, your task is to analyze the provided stock information and present a balanced view of the stock's upsides and downsides, followed by a final recommendation. Follow these steps:\n\n1. Identify Key Metrics:\n   - Review all provided metrics, including valuation, dividends, growth and performance, analyst opinions, financial health, profitability, market position, and risks.\n\n2. Categorize Upsides:\n   - List 4-5 positive aspects of the stock based on the data.\n   - For each upside, provide a brief explanation of why it's beneficial.\n\n3. Categorize Downsides:\n   - List 4-5 negative aspects or risks associated with the stock.\n   - Explain the potential impact of each downside on the stock's performance or investor sentiment.\n\n4. Analyze Overall Picture:\n   - Consider how the upsides and downsides balance each other.\n   - Evaluate the stock's current performance in the context of its potential future growth.\n\n5. Categorize Investment Horizon:\n   - Based on the analysis, categorize the stock as most suitable for:\n     a) Short-term investment (less than 1 year)\n     b) Medium-term investment (1-3 years)\n     c) Long-term investment (3+ years)\n   - Provide reasoning for this categorization, considering factors such as:\n     - Current valuation relative to growth prospects\n     - Dividend policy and yield\n     - Market trends and company's competitive position\n     - Potential catalysts for stock price movement in different timeframes\n\n6. Formulate Recommendation:\n   - Based on the analysis, provide one of the following recommendations: Strong Buy, Buy, Hold, Sell, or Strong Sell.\n   - Explain the reasoning behind your recommendation, considering different investor profiles (e.g., risk-averse vs. growth-oriented).\n\n7. Additional Considerations:\n   - Comment on how the stock might fit into a diversified portfolio.\n   - Mention any key metrics that investors should monitor going forward.\n\n8. Summarize:\n   - Provide a concise summary (2-3 sentences) of your analysis, including the recommended investment horizon and overall recommendation.\n\nRemember to maintain a balanced and objective tone throughout the analysis. Avoid using overly technical jargon, and explain any financial terms that may not be familiar to all readers. Your analysis should be thorough yet accessible to investors with varying levels of financial knowledge.\n",
                        f"Analyze the following stock: '{ticker}')",
                        f"Stock Evaluation:\n{json.dumps(stock_evaluation, indent=2)}"
                    ],
                },
            ]
        )

        response = chat_session.send_message("Provide the analysis based on the given instructions and stock evaluation. Use Markdown formatting for better readability.")
        analysis = response.text
        analysis_html = markdown2.markdown(analysis)

        # Store the original analysis in the session
        session['original_analysis'] = analysis

        return render_template('result.html', ticker=ticker, analysis=analysis_html, evaluation=stock_evaluation)
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    chat_history = session.get('chat_history', [])
    ticker = session.get('ticker', '')
    stock_evaluation = session.get('stock_evaluation', {})  # Retrieve stock evaluation from session

    # Configure Google AI
    genai.configure(api_key="AIzaSyDSYuKyOG8nQkNK4PNJqtIyrKJkfIeBgUQ")
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    print(stock_evaluation)
    # Prepare the initial context and chat history for the AI
    initial_context = [
        {
            "role": "user",
            "parts": [
                "# Instructions for Stock Analysis\n\nAs an LLM, your task is to analyze the provided stock information and present a balanced view of the stock's upsides and downsides, followed by a final recommendation. Follow these steps:\n\n1. Identify Key Metrics:\n   - Review all provided metrics, including valuation, dividends, growth and performance, analyst opinions, financial health, profitability, market position, and risks.\n\n2. Categorize Upsides:\n   - List 4-5 positive aspects of the stock based on the data.\n   - For each upside, provide a brief explanation of why it's beneficial.\n\n3. Categorize Downsides:\n   - List 4-5 negative aspects or risks associated with the stock.\n   - Explain the potential impact of each downside on the stock's performance or investor sentiment.\n\n4. Analyze Overall Picture:\n   - Consider how the upsides and downsides balance each other.\n   - Evaluate the stock's current performance in the context of its potential future growth.\n\n5. Categorize Investment Horizon:\n   - Based on the analysis, categorize the stock as most suitable for:\n     a) Short-term investment (less than 1 year)\n     b) Medium-term investment (1-3 years)\n     c) Long-term investment (3+ years)\n   - Provide reasoning for this categorization, considering factors such as:\n     - Current valuation relative to growth prospects\n     - Dividend policy and yield\n     - Market trends and company's competitive position\n     - Potential catalysts for stock price movement in different timeframes\n\n6. Formulate Recommendation:\n   - Based on the analysis, provide one of the following recommendations: Strong Buy, Buy, Hold, Sell, or Strong Sell.\n   - Explain the reasoning behind your recommendation, considering different investor profiles (e.g., risk-averse vs. growth-oriented).\n\n7. Additional Considerations:\n   - Comment on how the stock might fit into a diversified portfolio.\n   - Mention any key metrics that investors should monitor going forward.\n\n8. Summarize:\n   - Provide a concise summary (2-3 sentences) of your analysis, including the recommended investment horizon and overall recommendation.\n\nRemember to maintain a balanced and objective tone throughout the analysis. Avoid using overly technical jargon, and explain any financial terms that may not be familiar to all readers. Your analysis should be thorough yet accessible to investors with varying levels of financial knowledge.\n",
                f"Analyze the following stock: '{ticker}'",
                f"Stock Evaluation:\n{json.dumps(stock_evaluation, indent=2)}"
            ],
        },
        {
            "role": "model",
            "parts": [chat_history[0]["content"] if chat_history else ""]
        }
    ]
    
    # Add the rest of the chat history
    for msg in chat_history[1:]:
        initial_context.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]]
        })
    
    # Add the new user message
    initial_context.append({
        "role": "user",
        "parts": [message]
    })
    
    chat_session = model.start_chat(history=initial_context)
    print(initial_context)
    response = chat_session.send_message(f"You are an AI assistant specialized in analyzing the stock {ticker}. Please provide a concise and informative answer to the user's question, taking into account the context of the previous conversation.")
    answer = response.text
    answer_html = markdown2.markdown(answer)
    
    # Update the chat history
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    session['chat_history'] = chat_history
    
    return jsonify({'answer': answer_html})

if __name__ == '__main__':
    app.run(debug=True)
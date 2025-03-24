// src/App.js

import React, { useState } from 'react';
import axios from 'axios';
import StockSearch from './components/StockSearch';
import StockInfo from './components/StockInfo';
import Portfolio from './components/Portfolio';
import './App.css';

const App = () => {
    const [stock, setStock] = useState(null);
    const [portfolio, setPortfolio] = useState({});

    // Function to fetch stock data
    const fetchStockData = async (symbol) => {
        const apiKey = 'WW39XYHS0UHD2PVT'; // Replace with your API key
        const apiUrl = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${apiKey}`;

        try {
            const response = await axios.get(apiUrl);
            const data = response.data['Global Quote'];
            if (data) {
                setStock({
                    symbol: data['01. symbol'],
                    price: parseFloat(data['05. price']),
                    high: parseFloat(data['03. high']),
                    low: parseFloat(data['04. low'])
                });
            } else {
                setStock(null);
                alert('Stock data not found.');
            }
        } catch (error) {
            console.error('Error fetching stock data:', error);
            setStock(null);
            alert('Failed to fetch stock data. Please try again.');
        }
    };

    // Function to buy stock
    const buyStock = () => {
        if (stock && stock.symbol) {
            setPortfolio((prevPortfolio) => {
                const newPortfolio = { ...prevPortfolio };
                if (newPortfolio[stock.symbol]) {
                    newPortfolio[stock.symbol].shares += 1;
                } else {
                    newPortfolio[stock.symbol] = {
                        shares: 1,
                        price: stock.price
                    };
                }
                return newPortfolio;
            });
        }
    };

    // Function to sell stock
    const sellStock = () => {
        if (stock && stock.symbol && portfolio[stock.symbol]) {
            setPortfolio((prevPortfolio) => {
                const newPortfolio = { ...prevPortfolio };
                if (newPortfolio[stock.symbol].shares > 1) {
                    newPortfolio[stock.symbol].shares -= 1;
                } else {
                    delete newPortfolio[stock.symbol];
                }
                return newPortfolio;
            });
        }
    };

    return (
        <div className="App">
            <header>
                <h1>Stock Trading App</h1>
            </header>
            <StockSearch onSearch={fetchStockData} />
            <StockInfo stock={stock} />
            <div className="actions">
                <button onClick={buyStock} disabled={!stock}>Buy</button>
                <button onClick={sellStock} disabled={!stock}>Sell</button>
            </div>
            <Portfolio portfolio={portfolio} />
        </div>
    );
};

export default App;
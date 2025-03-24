// src/components/StockSearch.js

import React from 'react';

const StockSearch = ({ onSearch }) => {
    const [symbol, setSymbol] = React.useState('');

    const handleSearch = () => {
        if (symbol.trim() !== '') {
            onSearch(symbol.trim().toUpperCase());
        }
    };

    return (
        <div className="stock-search">
            <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                placeholder="Enter stock symbol (e.g., AAPL)"
            />
            <button onClick={handleSearch}>Search</button>
        </div>
    );
};

export default StockSearch;
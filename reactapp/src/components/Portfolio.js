// src/components/Portfolio.js

import React from 'react';

const Portfolio = ({ portfolio }) => {
    return (
        <div className="portfolio">
            <h2>Portfolio</h2>
            {Object.keys(portfolio).length > 0 ? (
                <ul>
                    {Object.entries(portfolio).map(([symbol, data]) => (
                        <li key={symbol}>
                            {symbol}: {data.shares} shares at ${data.price.toFixed(2)} each
                        </li>
                    ))}
                </ul>
            ) : (
                <p>No stocks bought yet.</p>
            )}
        </div>
    );
};

export default Portfolio;
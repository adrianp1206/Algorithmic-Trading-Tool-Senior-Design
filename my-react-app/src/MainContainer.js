// MainContainer.js
import React, { useEffect, useState } from "react";
import StockContainer from "./StockContainer";
import PortfolioContainer from "./PortfolioContainer";
import SearchBar from "./SearchBar";

function MainContainer() {
  const [stocks, setStocks] = useState([]);
  const [portfolio, setPortfolio] = useState([]);
  const [sortBy, setSortBy] = useState("Alphabetically");
  const [filterBy, setFilterBy] = useState("Tech");
  const [loading, setLoading] = useState(true); // Add loading state
  const [error, setError] = useState(null); // Add error state

  useEffect(() => {
    setLoading(true);
    fetch("http://localhost:3001/stocks")
      .then((r) => {
        if (!r.ok) {
          throw new Error("Network response was not ok");
        }
        return r.json();
      })
      .then((data) => {
        setStocks(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  function handleAddStock(stockToAdd) {
    const stockInPortfolio = portfolio.find(
      (stock) => stock.id === stockToAdd.id
    );
    if (!stockInPortfolio) {
      setPortfolio([...portfolio, stockToAdd]);
    }
  }

  function handleRemoveStock(stockToRemove) {
    setPortfolio((portfolio) =>
      portfolio.filter((stock) => stock.id !== stockToRemove.id)
    );
  }

  const sortedStocks = [...stocks].sort((stock1, stock2) => {
    if (sortBy === "Alphabetically") {
      return stock1.name.localeCompare(stock2.name);
    } else {
      return stock1.price - stock2.price;
    }
  });

  const filteredStocks = sortedStocks.filter(
    (stock) => stock.type === filterBy
  );

  if (loading) return <div>Loading stocks...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <SearchBar
        sortBy={sortBy}
        onChangeSort={setSortBy}
        filterBy={filterBy}
        onChangeFilter={setFilterBy}
      />
      <div className="row">
        <div className="col-8">
          <StockContainer stocks={filteredStocks} onAddStock={handleAddStock} />
        </div>
        <div className="col-4">
          <PortfolioContainer
            stocks={portfolio}
            onRemoveStock={handleRemoveStock}
          />
        </div>
      </div>
    </div>
  );
}

export default MainContainer;
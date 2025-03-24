import React from "react";

function SearchBar({ sortBy, onChangeSort, filterBy, onChangeFilter }) {
  // Handler for when the sort option changes
  function handleSortChange(event) {
    onChangeSort(event.target.value);
  }

  // Handler for when the filter option changes
  function handleFilterChange(event) {
    onChangeFilter(event.target.value);
  }

  return (
    <div>
      {/* Sort options */}
      <strong>Sort by:</strong>
      <label>
        <input
          type="radio"
          value="Alphabetically"
          name="sort"
          checked={sortBy === "Alphabetically"}
          onChange={handleSortChange}
        />
        Alphabetically
      </label>
      <label>
        <input
          type="radio"
          value="Price"
          name="sort"
          checked={sortBy === "Price"}
          onChange={handleSortChange}
        />
        Price
      </label>
      <br />

      {/* Filter options */}
      <label>
        <strong>Filter:</strong>
        <select onChange={handleFilterChange} value={filterBy}>
          <option value="Tech">Tech</option>
          <option value="Sportswear">Sportswear</option>
          <option value="Finance">Finance</option>
        </select>
      </label>
    </div>
  );
}

export default SearchBar;
// index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import Header from './Header';
import MainContainer from './MainContainer';
import './index.css'; // Optional: for custom styles if you create this file

// Define the App component to combine Header and MainContainer
function App() {
  return (
    <div>
      <Header />
      <MainContainer />
    </div>
  );
}

// Create a root and render the App component
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
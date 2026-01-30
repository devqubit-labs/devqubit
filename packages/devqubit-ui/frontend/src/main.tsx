/**
 * DevQubit UI Development Entry Point
 *
 * This file is used for local development only.
 * When used as a library, consumers import from index.ts.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import './styles/globals.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

import React, { useState } from 'react';
import './App.css';

function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeReview = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review })
      });
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      console.log("Received data:", data); // Debug output
      setResult(data);
    } catch (err) {
      setError('Failed to analyze review. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentClass = (sentiment) => {
    if (sentiment === 'POSITIVE' || sentiment === 'Positive') return 'sentiment-positive';
    if (sentiment === 'NEGATIVE' || sentiment === 'Negative') return 'sentiment-negative';
    return 'sentiment-neutral';
  };

  const formatConfidence = (confidence) => {
    // Check if confidence is already a string with a % sign
    if (typeof confidence === 'string' && confidence.includes('%')) {
      return confidence;
    }
    
    // If it's a number, format it
    if (typeof confidence === 'number') {
      return confidence.toFixed(2) + '%';
    }
    
    // Otherwise, just return it as is
    return confidence;
  };

  return (
    <div className="app-container">
      <h1 className="page-title">Aspect-Based Sentiment Analysis</h1>
      <div className="input-section">
        <p className="description">
          Enter a review to analyze its sentiment. The system will identify overall sentiment and 
          analyze specific aspects mentioned in the review (like food, service, location, etc.).
        </p>
        <textarea
          className="review-textarea"
          rows="5"
          placeholder="Enter your review (e.g., Food is good but location should be much better and services is not good)"
          value={review}
          onChange={(e) => setReview(e.target.value)}
        ></textarea>
        <button
          className="analyze-button"
          onClick={analyzeReview}
          disabled={loading || !review.trim()}
        >
          {loading ? 'Analyzing...' : 'Analyze Review'}
        </button>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      {loading && <div className="loading-indicator">Analyzing your review...</div>}
      
      {result && (
        <div className="results-container">
          <h2 className="results-title">Analysis Results</h2>
          
          <div className="overall-result">
            <h3>Overall Sentiment</h3>
            <div className="sentiment-badge-container">
              <span className={`sentiment-badge ${getSentimentClass(result.overall_sentiment || result.overall)}`}>
                {result.overall_sentiment || result.overall}
              </span>
              <span className="confidence">
                Confidence: {formatConfidence(result.overall_confidence || result.confidence)}
              </span>
            </div>
            {result.processing_time && (
              <div className="processing-time">
                Analysis completed in {result.processing_time.toFixed(2)} seconds
              </div>
            )}
          </div>
          
          <div className="aspects-container">
            <h3>Aspect-Based Analysis</h3>
            {result.aspects && (
              Array.isArray(result.aspects) ? (
                // Handle the original array format
                result.aspects.length > 0 ? (
                  <div className="aspect-cards">
                    {result.aspects.map((aspect, index) => (
                      <div key={index} className="aspect-card">
                        <div className="aspect-header">
                          <span className="aspect-name">{aspect.aspect.toUpperCase()}</span>
                          <span className={`aspect-sentiment ${getSentimentClass(aspect.sentiment)}`}>
                            {aspect.sentiment}
                          </span>
                        </div>
                        <div className="aspect-confidence">
                          Confidence: {formatConfidence(aspect.confidence)}
                        </div>
                        <div className="aspect-text">
                          "{aspect.text}"
                        </div>
                      </div>
                    ))}</div>
                ) : (
                  <p className="no-aspects">No specific aspects identified in this review.</p>
                )
              ) : (
                // Handle the new object format
                Object.keys(result.aspects).length > 0 ? (
                  <div className="aspect-cards">
                    {Object.entries(result.aspects).map(([aspectName, aspectData], index) => (
                      <div key={index} className="aspect-card">
                        <div className="aspect-header">
                          <span className="aspect-name">{aspectName.toUpperCase()}</span>
                          <span className={`aspect-sentiment ${getSentimentClass(aspectData.sentiment)}`}>
                            {aspectData.sentiment}
                          </span>
                        </div>
                        <div className="aspect-confidence">
                          Confidence: {formatConfidence(aspectData.confidence)}
                        </div>
                        <div className="aspect-text">
                          "{aspectData.text}"
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="no-aspects">No specific aspects identified in this review.</p>
                )
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
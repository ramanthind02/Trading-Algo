# General guidelines
- Simple strategies are better
	- Feature should have 3-4 flexible parameters max 
- Features should be taking advantage of some logical inefficiency 
	- Momentum, Mean reversion, seasonal tendency, volatility clustering, breakout, 
- A feature is synonymous with strategy, model, etc... 
## Boolean vs Continuous features 

### Continuous features
- Continuous features are judged based on their relationship to the target and its fixed holding period
	- Ex. RSI < 20, 1-day log returns tend to be more bullish
- We use binning to transform the feature into a discrete value afterwards
- These features are simple and robust
- Serve as the basic building blocks for engineering more complex features  

### Rule based features
- Rule based features output boolean/categorical outputs 
	- 1 = Hold long position, 0 = Flat, -1 = Hold short position
	- No confidence in predictions, discrete values
	- EX. Buy when rsi < 20, hold until rsi > 70
- This creates a complex mapping, the holding period may differ on each signal
- These features are pre-transformed into discrete values, no need for binning 
	- We can think of the binning is already built into the model parameters 
- Important to limit the number of rules to prevent over-fitting








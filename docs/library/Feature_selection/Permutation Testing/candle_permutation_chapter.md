This is the extracted text from the provided images, organized by page.

### Page 31
**Core Algorithms**
**31**

**Permuting Bars**

Permuting bars is significantly more difficult than permuting a single price series. A key tenet of permutation tests is that the statistical properties of the permuted series must be the same as the statistical properties of the original series. Otherwise, trading results obtained from a permuted series, whose predictable patterns have been destroyed, are not comparable to trading results from the unpermuted series. In other words, we must not introduce confounding factors into the test. Permutation must destroy predictable patterns without destroying anything else that impacts test results. We want our permutation test to evaluate performance with versus without predictability, but with all other factors held constant. That can be harder than it might seem. And this is not even a well defined requirement. Consider:

*   Despite our best efforts, permutation might damage some statistical property that is crucial to the trading system test but that we didn’t even think of.
*   We may go to great lengths to preserve some statistical property that is of no importance to the test.

Despite these somewhat discouraging thoughts, in my experience if we take just a few precautions we are covered for virtually every possible trading system test. In particular, we must preserve the distribution of intra-bar price relationships, and we must similarly preserve the distribution of inter-bar price relationships.

Intra-bar price relationships include the net move from open to close, the range between the high and the low, the maximum amount by which the price moves above the open, and the maximum amount by which the price moves below the open. In order to preserve the statistical distribution of these quantities, we can define any given bar by three quantities: The high minus the open, the low minus the open, and the close minus the open. Suppose we pass through the market history and compute these three quantities for each bar. Note that these are not absolute numbers; they are all relative to the open of the bar, whatever that may be. If we permute these triplets, these intra-bar distributions will remain unchanged.

---

### Page 32
**32**
**Core Algorithms**

The inter-bar relationships require some thought, because we can easily introduce unnatural price artifacts if we are not careful. Suppose we defined the inter-bar relationship as the change from one open to the next open. Consider a permutation in which a strong down bar (the close is much lower than the open) is followed by a bar with a high positive open-to-open. Then the close of the first bar would be very much lower than the open of the second bar, a situation that would almost never happen in real life.

In fact, bars nearly always open near the close of the prior bar, especially if they are intraday bars. This leads us to the quantity that we permute to vary inter-bar relationships. For each pair of bars we compute the change from the close of one bar to the open of the next bar. We then permute this array of inter-bar changes.

Just as we did for single prices, we want the open of the first bar to be equal for all permutations, and we also want the close of the last bar to be equal. This preserves any global trend, a feature that is crucial for many or most trading system tests. In fact, as we will see later when specific applications are discussed, it is good if the first bar in all permutations is identical to that of the original data.

This leads us to a simple method for reconstructing a bar series after the intra-bar and inter-bar relationships have been permuted. Begin with the first bar of the original data. Add to its close the first permuted value of the inter-bar series, the close-to-next-open differences. This gives us the open of the next bar. Then use the next permuted intra-bar triplet to give us the high, low, and close of this bar. The close of this bar plus the next permuted inter-bar difference provides the open of the following bar, and so forth.

This method of permuting bars preserves the distribution of all intra-bar and inter-bar relationships, also preserves the global trend, but completely destroys any predictable patterns in the market history. This algorithm should be usable for nearly any trading system test. However, do note the warning about redistribution of day range extremes discussed on Page 39. This small flaw may impact some trading systems.

---

### Page 33
**Core Algorithms**
**33**

Here is the class declaration for the BarPermute class. We will invoke the constructor, specifying the four price arrays that will serve as both input of the original bars and output of the permuted bars. As we did for price permutation, these four quantities are arrays of pointers. For example, `open[imarket]` is a pointer to the chronological open prices for the specified market. Also as we did for price permutation, index is the index in the price history of the basis bar, the bar that does not change under permutation and from which the permuted series is reconstructed.

```cpp
class BarPermute {

public:
    BarPermute (
        int np ,            // Number of prices
        int nmkt ,          // Number of markets
        int index ,         // Index of basis bar, one prior to first permuted price
        double **open ,     // Input of nmkt by nc opens
        double **high ,     // Input of nmkt by nc highs
        double **low ,      // Input of nmkt by nc lows
        double **close      // Input of nmkt by nc closes
        );
    ~BarPermute ();
    void do_permute ();

private:
    int ok ;                // Was memory allocation successful?
    int n_prices ;          // Number of prices (bars)
    int n_markets ;         // Number of markets
    int permute_index ;     // Index of first permuted price
    double *basis_open ;    // Work area for saving basis prices (bars)
    double *basis_high ;    // Ditto
    double *basis_low ;     // Ditto
    double *basis_close ;   // Ditto
    double **open_ptr ;     // Saves pointer to user's price input/output
    double **high_ptr ;     // Ditto
    double **low_ptr ;      // Ditto
    double **close_ptr ;    // Ditto
    double **rel_open ;     // Work area of n_markets arrays np long
    double **rel_high ;     // Ditto
    double **rel_low ;      // Ditto
    double **rel_close ;    // Ditto
};
```

---

### Page 34
**34**
**Core Algorithms**

The constructor copies the parameters to private areas. When it saves the index it adds one so that `permute_index` points to the first unpermuted bar.

```cpp
BarPermute::BarPermute (
    int np ,                // Number of prices
    int nmkt ,              // Number of markets
    int index ,             // Index of basis price, one prior to first permuted price
    double **open ,         // Input of nmkt by nc opens
    double **high ,         // Input of nmkt by nc highs
    double **low ,          // Input of nmkt by nc lows
    double **close          // Input of nmkt by nc closes
    )
{
    int i, iprice, imarket ;

    n_prices = np ;         // Copy parameters to private areas
    n_markets = nmkt ;
    open_ptr = open ;
    high_ptr = high ;
    low_ptr = low ;
    close_ptr = close ;
    permute_index = index + 1 ; // Point to first permuted price

    // The memory allocation code in PRICE_PERMUTE.CPP includes clean
    // failure in case of insufficient memory. That is omitted here for clarity.

    basis_open = (double *) malloc ( 4 * n_markets * sizeof(double) ) ;
    rel_open = (double **) malloc ( 4 * n_markets * sizeof(double *) ) ;

    basis_high = basis_open + n_markets ;     // We do just one allocation above, 4 times
    basis_low = basis_high + n_markets ;      // larger than needed, then split it here
    basis_close = basis_low + n_markets ;
    rel_high = rel_open + n_markets ;         // Do the same for the change arrays
    rel_low = rel_high + n_markets ;
    rel_close = rel_low + n_markets ;

    for (imarket=0 ; imarket < n_markets ; imarket++) { // Allocate changes for each market
        rel_open[imarket] = (double *) malloc ( 4 * n_prices * sizeof(double) ) ;
        rel_high[imarket] = rel_open[imarket] + n_prices ;
        rel_low[imarket] = rel_high[imarket] + n_prices ;
        rel_close[imarket] = rel_low[imarket] + n_prices ;
    }
```

---

### Page 35
**Core Algorithms**
**35**

For each market, do the following:

1)  Save the basis bar. This is necessary because the caller has almost certainly taken logs of prices before invoking the constructor, and therefore may exponentiate the permuted series. Since the same arrays serve as both input and output, exponentiation would destroy the basis bar.

2)  Compute `rel_open` as the difference between each bar’s open and the prior bar’s close. This is the *inter-bar* data discussed earlier.

3)  Compute `rel_high`, `rel_low`, and `rel_close` as the trio that defines the *intra-bar* behavior discussed earlier.

```cpp
    for (imarket=0 ; imarket < n_markets ; imarket++) {
        basis_open[imarket] = open[imarket][index] ;
        basis_high[imarket] = high[imarket][index] ;
        basis_low[imarket] = low[imarket][index] ;
        basis_close[imarket] = close[imarket][index] ;
        for (iprice=permute_index ; iprice < n_prices ; iprice++) {
            rel_open[imarket][iprice] = open[imarket][iprice] - close[imarket][iprice-1] ;
            rel_high[imarket][iprice] = high[imarket][iprice] - open[imarket][iprice] ;
            rel_low[imarket][iprice] = low[imarket][iprice] - open[imarket][iprice] ;
            rel_close[imarket][iprice] = close[imarket][iprice] - open[imarket][iprice] ;
        }
    }
}
```

We call `do_permute()` to return a permuted bar array. We must separately shuffle the inter-bar gaps and the intra-bar trios. In both cases we will be shuffling `n_prices - permute_index` terms. This code first shuffles the trios and then the gaps. The standard shuffling algorithm shown on Page 24 is used both times.

```cpp
void BarPermute::do_permute ()
{
    int i, j, iprice, imarket ;
    double dtemp ;

    i = n_prices - permute_index ; // Number remaining to be shuffled
```

---

### Page 36
**36**
**Core Algorithms**

```cpp
    while (i > 1) { // While at least 2 left to shuffle
        j = (int) (unifrand() * i) ;
        if (j >= i) // Should never happen, but be safe
            j = i - 1 ;
        --i ;
        for (imarket=0 ; imarket < n_markets ; imarket++) { // Shuffle the intra-bar trios
            dtemp = rel_high[imarket][i+permute_index] ;
            rel_high[imarket][i+permute_index] = rel_high[imarket][j+permute_index] ;
            rel_high[imarket][j+permute_index] = dtemp ;
            dtemp = rel_low[imarket][i+permute_index] ;
            rel_low[imarket][i+permute_index] = rel_low[imarket][j+permute_index] ;
            rel_low[imarket][j+permute_index] = dtemp ;
            dtemp = rel_close[imarket][i+permute_index] ;
            rel_close[imarket][i+permute_index] = rel_close[imarket][j+permute_index] ;
            rel_close[imarket][j+permute_index] = dtemp ;
        }
    } // Shuffle the intra-bar trios

    // Separately shuffle the close-to-open changes,
    // permuting each market the same to preserve correlations.

    i = n_prices - permute_index ; // Number remaining to be shuffled
    while (i > 1) { // While at least 2 left to shuffle
        j = (int) (unifrand() * i) ;
        if (j >= i) // Should never happen, but be safe
            j = i - 1 ;
        --i ;
        for (imarket=0 ; imarket < n_markets ; imarket++) {
            dtemp = rel_open[imarket][i+permute_index] ;
            rel_open[imarket][i+permute_index] = rel_open[imarket][j+permute_index] ;
            rel_open[imarket][j+permute_index] = dtemp ;
        }
    } // Shuffle the close-to-open changes
```

The last step is to rebuild the permuted series. The basis bar remains unchanged, so we recover it from where it was saved in the constructor call. We begin reconstruction at the bar immediately following the basis bar. The open of each new bar is the close of the prior bar (`close_ptr[imarket][iprice-1]`) plus the next permuted inter-bar gap (`rel_open[imarket][iprice]`). The high, low, and close of this new bar are all relative to the open of the bar.

---

### Page 37
**Core Algorithms**
**37**

```cpp
    for (imarket=0 ; imarket < n_markets ; imarket++) {
        open_ptr[imarket][permute_index-1] = basis_open[imarket] ; // Recover basis price
        high_ptr[imarket][permute_index-1] = basis_high[imarket] ;
        low_ptr[imarket][permute_index-1] = basis_low[imarket] ;
        close_ptr[imarket][permute_index-1] = basis_close[imarket] ;

        for (iprice=permute_index ; iprice < n_prices ; iprice++) { // Rebuild permuted series
            open_ptr[imarket][iprice] = close_ptr[imarket][iprice-1] + rel_open[imarket][iprice] ;
            high_ptr[imarket][iprice] = open_ptr[imarket][iprice] + rel_high[imarket][iprice] ;
            low_ptr[imarket][iprice] = open_ptr[imarket][iprice] + rel_low[imarket][iprice] ;
            close_ptr[imarket][iprice] = open_ptr[imarket][iprice] + rel_close[imarket][iprice] ;
        } // For iprice
    } // For imarket
} // End of do_permute()
```

**Permuting Intraday Data**

Intraday data can be represented as shown in Figure 2.1 below. We have a ‘basis’ day, followed by an overnight gap, then the first permuted day, another overnight gap, another day, and so forth. Each day can be composed of individual prices (ticks) or bars of any size.

[Diagram Title: Figure 2.1 Intraday price representation showing days and gaps]
[Box 1: Basis day]
[Box 2: First overnight gap]
[Box 3: First permuted day]
[Box 4: Second overnight gap]
[Box 5: Second permuted day]
[Box 6: Subsequent gaps / days]

---

### Page 38
**38**
**Core Algorithms**

I won’t provide specific code for permuting intraday data, because the code is highly dependent on how you store the information. However, the process is a simple extension of what we just saw for bar data. We permute the (log) prices or bars within each day, *separately for each day*, exactly as has been discussed in the prior sections. (We do not permute the prices/bars in the basis day.) We also compute the vector of overnight gaps, the difference between the closing price of one day and the opening price of the next day. This gap vector is permuted.

In order to rebuild the permuted dataset, it’s easiest if, separately for each day, we subtract the open of each day from all prices in that day, which of course leaves each day opening at a price of zero. Also, rather than trying to permute entire blocks of intraday data, we leave them in their original order and storage format, and instead permute an index vector that defines the order in which permuted days are appended. Then, to rebuild the permuted intraday data, we begin with the basis day, unchanged. Add to its close the first permuted overnight gap. This gives the amount to be added to each price within the next day. Repeat this process until all days have been included. Here is this algorithm stated more concisely. First, initialize:

1)  Compute the vector of overnight gaps (close of one day to open of the next day).
2)  Separately for each day except the basis day, subtract the open of that day from all prices in that day, including the open (thereby leaving the open at zero).
3)  Initialize an index vector of integers 0, 1, 2, .... This vector contains as many elements as there are days to be permuted.

Repeat as often as desired to create permutations:

1)  Separately for each day except the basis day, permute the data for that day using either the single price or the bar algorithm described in prior sections.
2)  Permute the vector of overnight gaps.

---

### Page 39
**Core Algorithms**
**39**

3)  Permute the vector of indices that were initialized in Step 3 above.
4)  Let the basis day be the first ‘permuted’ day.
5)  Select the day (already permuted in Step 1 above) identified by the first permuted index. Add to each of its prices the close of the basis day as well as the first permuted overnight gap.
6)  Select the day identified by the second permuted index. Add to each of its prices the close of the prior day as well as the second permuted overnight gap.
7)  Repeat Step 6 above until all days are appended.

In case you are wondering why each individual day must be permuted separately... Suppose we pooled the intraday changes into one big permutation pool and randomly selected our new daily data from this pool. This would tend toward homogeneity in daily ranges. As we built each day we would get some big jumps and some little jumps, and we would end up with each day having about the same net change. This is not representative of real-life daily action. By internally permuting each day separately we preserve the statistical distribution of daily net changes. And obviously the distribution of overnight gaps remains unchanged as well. Finally, I leave it as a simple exercise for the reader to confirm that the long-term trend of the market also remains unchanged.

This algorithm should perform well for many intraday trading systems. Its one apparent flaw is that changes in intraday volatility will be scattered across the permuted data instead of being clumped as is the usual situation. This may cause a problem for some trading systems. In other words, in real life we will have periods of days, weeks, or even months when day ranges are unusually high or low. However, when we shuffle across the entire historical time period we randomly distribute unusually high or low day ranges, which is unnatural. In my experience this is rarely, if ever a problem, but you should know about it. I am not aware of any practical fix.

---

### Page 40
**40**
**Core Algorithms**

**What About Night Sessions?**

Including overnight sessions is a trivial extension of the algorithm we just saw. In addition to separately permuting each day session (except the basis section), we also individually permute each night session. We also have to compute and permute *two* gaps, the night-close-to-day-open, and the day-close-to-night-open. We also have to initialize and subsequently permute *two* index vectors, one for selecting day sessions and one for selecting night sessions.

To rebuild a permuted price series, begin with the unchanged basis day session. Add to its close the first permuted day-to-night gap to get the open of the night session, which has been individually permuted already. Add to its close the first permuted night-to-day gap in order to get the open of the next day session. Repeat.
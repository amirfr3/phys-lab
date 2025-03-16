import numpy as np
import pandas as pd

# Discrete derivative helper function
def discrete_derivative(df: pd.DataFrame, columns: tuple, derivative: str, period: int = 1) -> pd.DataFrame:
  """
  Accept a pandas dataframe, and a list of columns (should be exactly 4,
   this is checked, can be pandas-compatible names). These should be 2 sets of
   data, the actual data points, and its uncertainty.
   The order is (x, delta_x, y, delta_y) It then does a discrete
   derivative of the second column over the first, so dy/dx. The user must also
   supply a name for the derivative. The output is a copy of the input dataframe
   with a column with the supplied derivative name containing the derivative,
   and another collumn named "delta_<name>" containing the calculated uncertainty.
   * Note 1: the index of the result fits the result of the input, assuming the
     input is sorted.
   * Note 2: We lose 2 datapoints because of how a discrete derivative is
     calculated, so there are Nan lines, it's up to the caller to handle these.
  """
  delta_derivative = f"delta_{derivative}"
  # Set first row to None
  res = df.copy()
  x, delta_x, y, delta_y = columns
  res[derivative] = df[y].diff(period) / df[x].diff(period)
  a, delta_a = df[y], df[delta_y]
  b, delta_b = df[y].shift(period), df[delta_y].shift(period)
  c, delta_c = df[x], df[delta_x]
  d, delta_d = df[x].shift(period), df[delta_x].shift(period)
  res[delta_derivative] = np.sqrt(((b*delta_a)/(c-d))**2 +
    ((a*delta_b)/(c-d))**2 + ((((a-b)/((c-d)**2))**2)*(delta_d**2 + delta_c**2)))
  return res

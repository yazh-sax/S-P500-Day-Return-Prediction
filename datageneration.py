import pandas as pd

class DayProfileLabeler:

    def __init__(self, bufferMult, trendRange, narrowRange):
        self.buffer_multiplier = bufferMult
        self.trend_range = trendRange
        self.narrow_fh_range = narrowRange

    def label_days(self, fh_high, fh_low, d_high, d_low, d_close):
        # Define ranges
        d_range = d_high - d_low
        fh_range = fh_high - fh_low
        buffer = self.buffer_multiplier * d_range
        fh_range_ratio = fh_range / d_range

        # Normal day
        if (0.5 <= fh_range_ratio <= 0.85) and (d_high - fh_high <= buffer) and (fh_low - d_low <= buffer):
            return "normal"

        # Normal variation day
        if (0.35 <= fh_range_ratio <= 0.65) and ((fh_low <= (d_low + buffer)) is not (fh_high >= (d_high + buffer))):
            return "normal_var"

        # Trend Day
        if (d_range >= self.trend_range) and ((d_close - d_low <= (0.25 * d_range)) or (d_high - d_close <= (0.25 * d_range))):
            return "trend"

        # Non-trend
        if (fh_range_ratio >= 0.9) and (d_high - fh_high <= buffer) and (fh_low - d_low <= buffer):
            return "non_trend"

        # Neutral
        if (d_high > fh_high) and (d_low < fh_low) and (fh_low <= d_close <= fh_high):
            return "neutral"

        # Narrow day
        if (fh_range <= self.narrow_fh_range) and (d_high > fh_high or d_low < fh_low):
            return "narrow"

        # Neutral extreme
        if (d_high > fh_high) and (d_low < fh_low) and ((d_close <= (fh_low - buffer)) or (d_close >= (fh_high + buffer))):
            return "neutral_ext"

        # Any other day
        return "other"


# Stationarize Data 
def to_stationary(days_data):
    days_data["FH_High"] = days_data["FH_High"] / days_data["D_High"]
    days_data["FH_Low"] = days_data["FH_Low"] / days_data["D_High"]
    days_data["D_Low"] = (days_data["D_Low"] / days_data["D_High"])
    days_data["D_Close"] = days_data["D_Close"] / days_data["D_High"]
    days_data["D_High"] = 1.00
    return days_data

def label_days(days_data):
    # Declaring lists which will be used as columns
    labels = []

    # Initializing labeler
    labeler = DayProfileLabeler(0.1, 50, 10)

    # Iterator
    for index, row in days_data.iterrows():
        current_day = labeler.label_days(row["FH_High"], row["FH_Low"], row["D_High"], row["D_Low"], row["D_Close"])

        if (current_day == "normal"):
            labels.append(0)
        if (current_day == "normal_var"):
            labels.append(1)
        if (current_day == "trend"):
            labels.append(2)
        if (current_day == "non_trend"):
            labels.append(3)
        if (current_day == "neutral"):
            labels.append(4)
        if (current_day == "narrow"):
            labels.append(5)
        if (current_day == "neutral_ext"):
            labels.append(6)
        if (current_day == "other"):
            labels.append(7)

    # Append to dataframe
    days_data['labels'] = labels
    return days_data

# Create first hour and daily data from five minute data
def create_5m_Data(file_name):
    # Parameters
    open_time = "06:30:00"
    fh_close_time = "07:30:00"
    d_close_time = "13:00:00"

    # CSV to Pandas Dataframe
    data_file = pd.read_csv(file_name)

    # Output placeholders
    results = pd.DataFrame(columns=['Date', 'FH_High', 'FH_Low', 'D_BidVol', 'D_AskVol', 'D_High', 'D_Low', 'D_Close',
                                    'D_BidVol', 'D_AskVol'])
    date = data_file['Date'][0]
    fh_high, fh_low, fh_bidvol, fh_askvol, d_high, d_low, d_close, d_bidvol, d_askvol = 0.00, float(
        'inf'), 0, 0, 0.00, float('inf'), 0.00, 0, 0

    # Toggles
    d_toggle = False
    fh_toggle = False

    # Iterator
    for index, row in data_file.iterrows():
        if row["Date"] != date:
            if 0 not in [fh_high, fh_low, fh_bidvol, fh_askvol, d_high, d_low, d_close, d_bidvol, d_askvol]:
                # write to results dataframe
                temp_results = [date, fh_high, fh_low, fh_bidvol, fh_askvol, d_high, d_low, d_close, d_bidvol, d_askvol]
                results.loc[len(results.index) + 1] = temp_results
            # reset values
            date = row["Date"]
            fh_high, fh_low, fh_bidvol, fh_askvol, d_high, d_low, d_close, d_bidvol, d_askvol = 0.00, float(
                'inf'), 0, 0, 0.00, float('inf'), 0.00, 0, 0

        # Setting toggles and close value
        if row["Time"] == open_time:
            d_toggle = True
            fh_toggle = True
        if row["Time"] == fh_close_time:
            fh_toggle = False
        if row["Time"] == d_close_time:
            d_close = row["Open"]
            d_toggle = False

        # Setting parameters
        if fh_toggle is True:
            if row["High"] > fh_high:
                fh_high = row["High"]
            if row["Low"] < fh_low:
                fh_low = row["Low"]
            fh_bidvol += row["BidVolume"]
            fh_askvol += row["AskVolume"]

        if d_toggle is True:
            if row["High"] > d_high:
                d_high = row["High"]
            if row["Low"] < d_low:
                d_low = row["Low"]
            d_bidvol += row["BidVolume"]
            d_askvol += row["AskVolume"]

    return results

data = create_5m_Data("../dataProcessing/ES500DFootprintData.csv")
data = label_days(data)
data = to_stationary(data)
data.to_csv("profile_data.csv", encoding='utf-8', index=False)


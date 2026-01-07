import pandas as pd

listings = pd.read_csv('listings.csv')
calendar = pd.read_csv('calendar.csv', nrows=10000)

print("Listings IDs head:")
print(listings['id'].head().tolist())
print(f"Listings ID dtype: {listings['id'].dtype}")

print("\nCalendar IDs head:")
print(calendar['listing_id'].head().tolist())
print(f"Calendar ID dtype: {calendar['listing_id'].dtype}")

# Check intersection
l_ids = set(listings['id'].unique())
c_ids = set(calendar['listing_id'].unique())

common = l_ids.intersection(c_ids)
print(f"\nCommon IDs count (in sample): {len(common)}")
if len(common) > 0:
    print(f"Sample common ID: {list(common)[0]}")
else:
    print("No intersection found in first 10000 calendar rows.")
    
# Check full intersection (only if sample failed)
print("Checking full calendar intersection...")
calendar_full = pd.read_csv('calendar.csv', usecols=['listing_id'])
c_ids_full = set(calendar_full['listing_id'].unique())
common_full = l_ids.intersection(c_ids_full)
print(f"Common IDs count (full): {len(common_full)}")

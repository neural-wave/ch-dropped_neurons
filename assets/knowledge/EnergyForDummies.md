# Physics, Engineering, and Market Facts for Reasoning About PV Detection

## Physics and Energy

1. **Energy and power are different.** kWh is energy, the amount used over time. kW is power, the rate of use. A 15-minute reading of 0.5 kWh means an average power of 2 kW for that interval.
2. **Solar output is roughly a bell curve over the day.** It is near zero before dawn and after dusk, and peaks around solar noon. Solar noon does not match clock noon. In Switzerland it shifts by up to about an hour across the year and by the local longitude within the time zone.
3. **Solar output depends on season and latitude.** A given system produces several times more energy in June than in December at Swiss latitudes.
4. **Clouds reduce output sharply but not to zero.** A passing cloud can cut output by half or more within minutes, so noisy daytime patterns are normal.
5. **Panels are rated at standard test conditions.** kWp is output under 1000 W/m² irradiance at 25 °C cell temperature. Real yields in Switzerland are commonly around 900 to 1100 kWh per kWp per year, so a 10 kWp system typically produces roughly 9,000 to 11,000 kWh annually. Use this as a sanity check on any capacity estimate.
6. **Heat reduces efficiency.** Panels lose output as they warm, so summer peaks are less than the linear irradiance trend would suggest.
7. **Self-consumption depends on timing.** A household uses a larger share of its own solar energy when it runs appliances during the day. The rest is exported to the grid. The meter sees only the net.
8. **Inverters can clip.** If the inverter is smaller than the panel array, output is capped on bright days, which flattens the midday peak.
9. **Panel orientation changes the shape.** South-facing panels peak around midday. East or west facing panels shift the peak earlier or later and produce a flatter curve.
10. **Shading and tilt matter.** Partial shade or a steep roof changes the daily and seasonal profile, so capacity is not the only factor shaping the dip.

## Engineering and Metering

11. **Net metering hides gross production.** The consumption series is grid withdrawal, which equals load minus PV self-consumption. Gross load and PV output are never seen separately.
12. **Negative flows are not in this dataset.** Export to the grid would appear as negative net consumption in some meters. The dataset records withdrawals, so a midday dip reflects reduced withdrawal, and the data may not include export values at all.
13. **Smart meter readings are interval averages.** Each 15-minute value summarises the energy in that window, so brief spikes are smoothed.
14. **Time stamps need care.** Local time repeats in autumn and skips in spring. Epoch time is unambiguous.

## Energy Market and Context

15. **Swiss utilities bill by energy, not capacity.** Customers pay for kWh drawn, so a PV system reduces the bill through lower withdrawals. Feed-in tariffs for exported energy are a separate commercial matter.
16. **Commissioning and grid registration are separate events.** A system may be built before it is registered or switched on, so the "installation date" in a database may not match when the consumption changed.
17. **Extensions add capacity in steps.** Each extension raises total kWp, so the dip deepens at the extension date. The labelled capacity after an extension is the total, not the increment.

## Interpretation Rules to Avoid Misconceptions

- A dip in daytime use is not proof of PV. Check it against seasonal and holiday patterns.
- A capacity estimate far outside roughly 0.9 to 1.1 MWh per kWp per year of implied production is suspect.
- The commissioning date is when the signal begins, which can lag the physical installation.
- Customers with large self-consumption can look almost unchanged, so absence of a signal does not always mean no PV.
`legacy.xls` is a small BIFF8 workbook generated once with `xlwt` for the legacy
Excel-reader fixture. It contains a `Sales 2025` sheet with one formatted date,
one `0.0%` percentage, and a numeric total. The binary is checked in so tests do
not need `xlwt` at runtime.

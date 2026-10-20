# Documentation-derived ASE semantics (cases marked `review: true`)

The expected behaviour of these cases was **not** obtained by running the source on a live SAP ASE
server. It was taken from the Sybase/SAP ASE documentation, assuming default server and session
options (`ansinull off`, unchained transaction mode, default sort order). Paper claims for these
cases are therefore claims of agreement with a documentation-derived reference. Each source below
was checked against the reference translation; links are to the official Sybase infocenter.

| Case | Behaviour the reference relies on | Official source |
|---|---|---|
| C02 | `*=` keeps every row of the outer (first) table that meets the statement's restrictions; the inner table only supplies values when the join condition matches, so a restriction on the inner table limits matches, not outer rows | Transact-SQL User's Guide, "Transact-SQL outer joins": https://infocenter.sybase.com/help/topic/com.sybase.dc32300_1251/html/sqlug/X19290.htm |
| C06, C09, C27 | "A string concatenated with NULL evaluates to the value of the string" (an exception to the SQL standard) | Reference Manual: Building Blocks, "String concatenation operator": https://infocenter.sybase.com/help/topic/com.sybase.dc36271_1251/html/blocks/blocks254.htm |
| C10 | `group by` extension: extended (non-grouped) columns may appear in the select list; each row is returned with its group aggregate | Transact-SQL User's Guide, "Referencing other columns in queries using group by": https://infocenter-archive.sybase.com/help/topic/com.sybase.dc32300_1250/html/sqlug/sqlug115.htm |
| C19 | The empty string is always stored as a single space in variables and column data (policy case, scored on G1/G2 only) | Transact-SQL User's Guide 15.7, "Using the NULL character string": https://infocenter.sybase.com/help/topic/com.sybase.infocenter.dc32300.1570/html/sqlug/sqlug258.htm |
| C26 | With `ansinull off` (the default), `if @v = @i` is true when both are NULL; comparing an expression to an expression with `=` returns TRUE for nulls | Transact-SQL User's Guide, "Variables and null values" (Table 13-2): https://infocenter-archive.sybase.com/help/topic/com.sybase.dc32300_1250/html/sqlug/sqlug562.htm ; default `ansinull off`: Transact-SQL User's Guide 15.7, "case expressions and set ansinull": https://infocenter.sybase.com/help/topic/com.sybase.infocenter.dc32300.1570/html/sqlug/sqlug592.htm |
| C28 | Scalar aggregate with extended columns and no `group by`: every row of the table is returned with the whole-table aggregate | Transact-SQL User's Guide, "Using aggregates without group by": https://infocenter.sybase.com/help/topic/com.sybase.dc32300_1251/html/sqlug/sqlug121.htm |
| C30 | `*=` "selects all rows from the first table that meet the statement's restrictions", so a restriction on the outer table filters outer rows | Transact-SQL User's Guide, "Transact-SQL outer joins" (as C02) |
| C32 | `str_replace` "accepts NULL in the third parameter and treats it as an attempt to replace string_expression2 with NULL, effectively turning str_replace into a 'string cut' operation" | Reference Manual: Building Blocks 15.0, "str_replace": https://infocenter-archive.sybase.com/help/topic/com.sybase.help.ase_15.0.blocks/html/blocks/blocks213.htm |
| C33 | `inttohex` returns an 8-character hexadecimal string without a `0x` prefix. **Not documented: letter case.** The reference assumes upper case | Transact-SQL User's Guide, "Converting hexadecimal data": https://infocenter-archive.sybase.com/help/topic/com.sybase.dc32300_1250/html/sqlug/X82619.htm |
| C34 | `hextoint("0x00000100")` returns 256 (this exact input is the manual's example) | as C33 |
| C47 | Combines C26 and C30 rules with `raiserror` argument substitution | as C26, C30 |
| C51 | Combines the C10 `group by` extension with `= NULL` and `str_replace` | as C10, C26, C32 |

Notes found while checking the sources:

* Sybase itself calls the `*=` syntax ambiguous in some cases and recommends ANSI joins
  (ASE 12.5 "What's New": "Transact-SQL syntax in previous releases was ambiguous in some cases").
* The empty-string rule (C19) also affects any input that passes `''`. Case C13 originally had a
  call with an empty-string argument; it was replaced because its expected result depends on the
  C19 policy decision rather than on `str_replace`.
* The `group by` extension ignores `where` for extended-column rows. No corpus case combines the two.

One reference translation (C06) was corrected after an audit found that it contradicted the NULL
concatenation rule above; see the paper's threats-to-validity section.

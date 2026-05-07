from flask import Flask, request, jsonify, render_template
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

app = Flask(__name__)

# ─────────────────────────────────────────────
# DEFAULT DATASET
# ─────────────────────────────────────────────

DEFAULT_TRANSACTIONS = [
    
    ["bread", "milk", "butter", "eggs"],
    ["bread", "milk", "diapers", "beer"],
    ["milk", "diapers", "beer", "cola"],
    ["bread", "butter", "beer"],
    ["bread", "milk", "diapers", "butter"],
    ["milk", "butter", "eggs", "cheese"],
    ["bread", "diapers", "beer", "cola"],
    ["bread", "milk", "eggs", "cheese"],
    ["butter", "eggs", "cheese", "yogurt"],
    ["bread", "milk", "butter", "diapers"],
    ["diapers", "beer", "cola", "chips"],
    ["milk", "butter", "eggs", "yogurt"],
    ["bread", "cheese", "eggs", "butter"],
    ["milk", "beer", "cola", "chips"],
    ["bread", "milk", "cola", "chips"],
]

# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

# ─────────────────────────────────────────────
# LOAD DEFAULT DATA
# ─────────────────────────────────────────────

@app.route("/api/default_transactions")
def default_transactions():

    lines = [",".join(t) for t in DEFAULT_TRANSACTIONS]

    return jsonify({
        "transactions": "\n".join(lines)
    })

# ─────────────────────────────────────────────
# ASSOCIATION RULE MINING
# ─────────────────────────────────────────────

@app.route("/api/mine", methods=["POST"])
def mine():

    try:

        data = request.get_json()

        min_support = float(data.get("min_support", 0.05))
        min_confidence = float(data.get("min_confidence", 0.3))

        raw_txns = data.get("transactions", "")

        # ─────────────────────────────────────
        # PARSE TRANSACTIONS
        # ─────────────────────────────────────

        if raw_txns.strip():

            transactions = []

            for row in raw_txns.strip().splitlines():

                items = [
                    item.strip().lower()
                    for item in row.split(",")
                    if item.strip()
                ]

                # Remove duplicates
                items = list(set(items))

                if len(items) > 0:
                    transactions.append(items)

        else:
            transactions = DEFAULT_TRANSACTIONS

        # Validation
        if len(transactions) < 2:

            return jsonify({
                "error": "Minimum 2 transactions required"
            }), 400

        # ─────────────────────────────────────
        # TRANSACTION ENCODING
        # ─────────────────────────────────────

        te = TransactionEncoder()

        te_array = te.fit(transactions).transform(transactions)

        df = pd.DataFrame(
            te_array,
            columns=te.columns_
        )

        # ─────────────────────────────────────
        # APRIORI ALGORITHM
        # ─────────────────────────────────────

        frequent_itemsets = apriori(
            df,
            min_support=min_support,
            use_colnames=True,
            low_memory=True
        )

        # No frequent itemsets
        if frequent_itemsets.empty:

            return jsonify({
                "error": "No frequent itemsets found. Lower support value."
            }), 400

        # Add count
        frequent_itemsets["count"] = (
            frequent_itemsets["support"] * len(transactions)
        ).astype(int)

        # ─────────────────────────────────────
        # ASSOCIATION RULES
        # ─────────────────────────────────────

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        # No rules found
        if rules.empty:

            return jsonify({
                "error": "No association rules found. Lower confidence value."
            }), 400

        # ─────────────────────────────────────
        # FORMAT FREQUENT ITEMSETS
        # ─────────────────────────────────────

        freq_json = []

        for _, row in frequent_itemsets.iterrows():

            freq_json.append({

                "itemset": sorted(list(row["itemsets"])),

                "support": round(
                    float(row["support"]), 4
                ),

                "count": int(row["count"])
            })

        # ─────────────────────────────────────
        # FORMAT RULES
        # ─────────────────────────────────────

        rules_json = []

        for _, row in rules.iterrows():

            rules_json.append({

                "antecedent": sorted(
                    list(row["antecedents"])
                ),

                "consequent": sorted(
                    list(row["consequents"])
                ),

                "support": round(
                    float(row["support"]), 4
                ),

                "confidence": round(
                    float(row["confidence"]), 4
                ),

                "lift": round(
                    float(row["lift"]), 4
                )
            })

        # Sort rules by confidence
        rules_json = sorted(
            rules_json,
            key=lambda x: x["confidence"],
            reverse=True
        )

        # Unique items
        items = sorted(
            set(
                i
                for t in transactions
                for i in t
            )
        )

        # ─────────────────────────────────────
        # RESPONSE
        # ─────────────────────────────────────

        return jsonify({

            "num_transactions": len(transactions),

            "num_items": len(items),

            "num_frequent": len(freq_json),

            "num_rules": len(rules_json),

            "frequent_itemsets": freq_json[:200],

            "rules": rules_json[:100]
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ─────────────────────────────────────────────

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
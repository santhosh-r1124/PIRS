"""Ticket text for the SYNTHETIC generator: per-module vocabulary, per-category templates.

Wording is deliberately varied (several templates per category, module-specific objects,
transaction codes, jobs and interfaces, random prefixes, vague tickets) so that the
resolution recommender in Phase 7 cannot just match one keyword per category.
"""

from __future__ import annotations

import numpy as np

CATEGORIES = [
    "job_failure", "interface_idoc", "authorisation", "performance",
    "master_data", "posting_error", "output_print", "transport_change",
]  # fmt: skip

# Category mix per module (same order as CATEGORIES); rows are normalised when used.
CATEGORY_MIX = {
    "FI":    [0.15, 0.10, 0.12, 0.05, 0.15, 0.30, 0.08, 0.05],
    "CO":    [0.20, 0.02, 0.12, 0.06, 0.20, 0.30, 0.05, 0.05],
    "SD":    [0.08, 0.20, 0.10, 0.05, 0.17, 0.15, 0.20, 0.05],
    "MM":    [0.10, 0.18, 0.10, 0.05, 0.22, 0.20, 0.10, 0.05],
    "PP":    [0.20, 0.05, 0.08, 0.07, 0.25, 0.25, 0.05, 0.05],
    "WM":    [0.10, 0.20, 0.08, 0.05, 0.15, 0.25, 0.12, 0.05],
    "QM":    [0.05, 0.05, 0.10, 0.05, 0.30, 0.30, 0.10, 0.05],
    "PM":    [0.10, 0.05, 0.12, 0.05, 0.30, 0.25, 0.08, 0.05],
    "HCM":   [0.15, 0.05, 0.25, 0.05, 0.25, 0.15, 0.05, 0.05],
    "BASIS": [0.30, 0.15, 0.10, 0.30, 0.00, 0.00, 0.05, 0.10],
}  # fmt: skip

VOCAB = {
    "FI": {
        "obj": [
            "vendor invoice",
            "G/L posting",
            "payment run",
            "bank statement",
            "asset depreciation",
            "customer payment",
            "foreign currency valuation",
        ],
        "tx": ["FB60", "FB50", "F110", "FF_5", "AFAB", "F-28", "FAGL_FCV"],
        "job": ["F110_PAYMENT_RUN", "FI_BANK_STMT_IMPORT", "AFAB_DEPRECIATION", "FAGL_FCV_RUN"],
        "iface": ["bank interface", "electronic bank statement import", "payment medium file"],
    },
    "CO": {
        "obj": [
            "cost centre",
            "internal order settlement",
            "assessment cycle",
            "profitability report",
            "activity allocation",
        ],
        "tx": ["KSU5", "KO88", "KE30", "KB21N", "KP06"],
        "job": ["CO_SETTLEMENT_KO8G", "CO_ASSESSMENT_RUN", "COPA_REALIGNMENT"],
        "iface": ["CO-PA extractor", "planning upload"],
    },
    "SD": {
        "obj": [
            "sales order",
            "billing document",
            "outbound delivery",
            "pricing condition",
            "credit block",
            "customer master",
        ],
        "tx": ["VA01", "VF01", "VL02N", "VK11", "VKM3", "XD02"],
        "job": ["SD_BILLING_DUE_LIST", "SD_DELIVERY_DUE_LIST", "SD_CREDIT_RECHECK"],
        "iface": ["ORDERS IDoc", "INVOIC IDoc", "EDI order inbound"],
    },
    "MM": {
        "obj": [
            "purchase order",
            "goods receipt",
            "invoice verification",
            "material master",
            "MRP run",
            "stock transfer",
        ],
        "tx": ["ME21N", "MIGO", "MIRO", "MM02", "MD01", "MB1B"],
        "job": ["MRP_TOTAL_PLANNING", "MM_PO_OUTPUT", "MM_GR_IR_CLEARING"],
        "iface": ["ORDRSP IDoc", "DESADV IDoc", "supplier portal interface"],
    },
    "PP": {
        "obj": [
            "production order",
            "bill of material",
            "routing",
            "order confirmation",
            "planned order",
        ],
        "tx": ["CO01", "CS02", "CA02", "CO11N", "MD04"],
        "job": ["PP_ORDER_RELEASE", "PP_BACKFLUSH", "PP_CAPACITY_LEVELLING"],
        "iface": ["MES confirmation interface", "LOIPRO IDoc"],
    },
    "WM": {
        "obj": ["transfer order", "storage bin", "putaway", "picking", "physical inventory"],
        "tx": ["LT03", "LS02N", "LT12", "LI11N", "LX03"],
        "job": ["WM_TO_AUTO_CREATE", "WM_BIN_REORG"],
        "iface": ["WMTORD IDoc", "RF scanner interface"],
    },
    "QM": {
        "obj": [
            "inspection lot",
            "usage decision",
            "quality notification",
            "inspection plan",
            "results recording",
        ],
        "tx": ["QA32", "QA11", "QM01", "QP02", "QE51N"],
        "job": ["QM_LOT_CREATION", "QM_AUTO_UD"],
        "iface": ["LIMS interface", "QALS IDoc"],
    },
    "PM": {
        "obj": [
            "maintenance order",
            "maintenance notification",
            "maintenance plan",
            "equipment master",
            "functional location",
        ],
        "tx": ["IW31", "IW21", "IP10", "IE02", "IL02"],
        "job": ["PM_SCHEDULING_IP30", "PM_ORDER_SETTLEMENT"],
        "iface": ["mobile maintenance app", "SCADA interface"],
    },
    "HCM": {
        "obj": [
            "payroll run",
            "time evaluation",
            "employee master",
            "payroll posting",
            "leave request",
        ],
        "tx": ["PC00_M99_CALC", "PT60", "PA30", "PC00_M99_CIPE", "PT_ARQ"],
        "job": ["PAYROLL_SIMULATION", "TIME_EVAL_RPTIME00", "PAYROLL_POSTING"],
        "iface": ["HR master data IDoc", "bank transfer file", "time clock interface"],
    },
    "BASIS": {
        "obj": [
            "application server",
            "background work process",
            "RFC destination",
            "transport import",
            "system memory",
            "spool",
        ],
        "tx": ["SM37", "SM59", "STMS", "ST22", "SM21", "SP01"],
        "job": ["SAP_REORG_JOBS", "SAP_COLLECTOR_FOR_PERFMONITOR", "RDDIMPDP", "SAP_REORG_SPOOL"],
        "iface": ["RFC connection to PI/PO", "SAP Cloud Connector", "SMTP mail relay"],
    },
}

DESCRIPTIONS = {
    "job_failure": [
        "Background job {job} cancelled",
        "{job} terminated with error in {tx}",
        "Overnight batch {job} did not complete",
        "Job {job} failed - {obj} not processed",
        "Scheduled job {job} stuck in active status",
    ],
    "interface_idoc": [
        "{iface} errors - documents stuck in status 51",
        "IDocs failing for {obj} from {iface}",
        "Inbound {iface} not processed",
        "{iface} messages hanging in the queue",
        "Outbound {iface} not sent to partner",
    ],
    "authorisation": [
        "User cannot open {tx} - no authorisation",
        "Missing authorisation for {obj} in {tx}",
        "{tx} access denied for the team",
        "Authorisation error when changing {obj}",
        "Role missing for {obj} approval",
    ],
    "performance": [
        "{tx} running very slow",
        "Timeout when processing {obj}",
        "Short dump TIME_OUT in {tx}",
        "System response slow for {obj} users",
        "Report on {obj} takes hours to finish",
    ],
    "master_data": [
        "Incorrect {obj} data blocking processing",
        "{obj} missing mandatory field",
        "Wrong unit of measure on {obj}",
        "Duplicate {obj} records created",
        "{obj} not extended to plant / company code",
    ],
    "posting_error": [
        "{obj} cannot be posted - period closed",
        "Posting error in {tx}: account determination missing",
        "{obj} posting rejected - balance not zero",
        "Error 'tolerance exceeded' when posting {obj}",
        "Document for {obj} not created in {tx}",
    ],
    "output_print": [
        "{obj} output not printed",
        "Form for {obj} shows wrong layout",
        "Output for {obj} not triggered",
        "Print queue stuck for {obj}",
        "E-mail output of {obj} not received by partner",
    ],
    "transport_change": [
        "Error in {tx} after yesterday's transport",
        "{obj} processing broken after change deployment",
        "Regression in {obj} after release",
        "New configuration causing errors in {tx}",
        "Custom code change dumps in {tx}",
    ],
}

RESOLUTIONS = {
    "job_failure": [
        "Corrected job variant and restarted {job} via SM37.",
        "Batch user was missing an authorisation; fixed and rescheduled {job}.",
        "Cleared lock entries in SM12 and re-ran {job}.",
        "Raised runtime limit and re-ran {job} successfully.",
        "Fixed input data for {obj} and restarted {job}.",
    ],
    "interface_idoc": [
        "Corrected partner profile in WE20 and reprocessed IDocs with BD87.",
        "Fixed field mapping for {obj} and reprocessed the failed IDocs.",
        "Removed blocking entry in SMQ2 and restarted the queue.",
        "Updated RFC destination and resent {iface} messages.",
        "Maintained missing conversion entry and reprocessed messages.",
    ],
    "authorisation": [
        "Analysed SU53 trace and added the missing authorisation object to the role.",
        "Assigned the correct role in PFCG after manager approval.",
        "Corrected organisational values in the role and regenerated the profile.",
        "Granted temporary firefighter access; permanent role change requested.",
        "Fixed role mapping in GRC and ran user comparison (PFUD).",
    ],
    "performance": [
        "Updated table statistics and added a secondary index for the selection.",
        "Cancelled the long-running work process and restarted the transaction.",
        "Optimised report selection by adding a date filter for {obj}.",
        "Increased work-process memory parameters after ST22 analysis.",
        "Archived old {obj} data to reduce table size.",
    ],
    "master_data": [
        "Corrected {obj} master data and reprocessed the document.",
        "Extended {obj} to the required organisational level.",
        "Merged duplicate records and flagged the obsolete one for deletion.",
        "Updated the field by mass change and informed the data owner.",
        "Fixed data in the source system and re-sent the master record.",
    ],
    "posting_error": [
        "Opened the posting period in OB52 for the affected account type.",
        "Maintained the missing account determination entry and reposted.",
        "Adjusted tolerance group and reposted {obj}.",
        "Corrected the posting date and reposted the document.",
        "Reversed the incorrect document and posted a correction.",
    ],
    "output_print": [
        "Maintained output condition record and re-triggered the output.",
        "Corrected the form template and reprinted.",
        "Restarted the output device / spool in SPAD.",
        "Fixed e-mail address in partner master and re-sent the output.",
        "Reprocessed the output from the message status screen.",
    ],
    "transport_change": [
        "Rolled back the transport via STMS; fix scheduled for next release.",
        "Imported the missing dependent transport.",
        "Corrected the configuration in a follow-up transport.",
        "Applied hot-fix to custom code and retested.",
        "Reverted the change and raised a defect with the project team.",
    ],
}

VAGUE = [
    "Issue with {obj}",
    "{tx} not working",
    "Users report problem in {tx}",
    "Error message when processing {obj}",
    "Unable to complete {obj}",
]
PREFIXES = ["", "", "", "", "URGENT: ", "Users report: ", "Helpdesk: ", "Follow-up: "]
RESOLUTION_SUFFIXES = [
    "", "", " User confirmed the fix.", " Monitored for 24h, no recurrence.",
    " Root cause added to knowledge base.",
]  # fmt: skip


def _slots(module: str, rng: np.random.Generator) -> dict[str, str]:
    """One object / transaction / job / interface per ticket, shared by text and notes."""
    v = VOCAB[module]
    return {k: v[k][rng.integers(len(v[k]))] for k in ("obj", "tx", "job", "iface")}


def _pick(options: list[str], rng: np.random.Generator) -> str:
    return options[rng.integers(len(options))]


def describe(
    module: str,
    category: str,
    rng: np.random.Generator,
    vague_share: float = 0.1,
    source_module: str | None = None,
) -> tuple[str, str]:
    """Return (short_description, resolution_notes) for one synthetic ticket."""
    slots = _slots(module, rng)
    pool = VAGUE if rng.random() < vague_share else DESCRIPTIONS[category]
    body = _pick(pool, rng).format(**slots)
    text = _pick(PREFIXES, rng) + body[0].upper() + body[1:]
    if source_module and rng.random() < 0.4:  # some tickets mention the upstream module
        text += f" (possibly related to {source_module} issue)"
    notes = _pick(RESOLUTIONS[category], rng).format(**slots) + _pick(RESOLUTION_SUFFIXES, rng)
    return text, notes

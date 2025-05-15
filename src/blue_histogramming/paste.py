groups = {
    "groups": [
        {
            "name": "entry",
            "type": "Group",
            "items": [{"name": "NX_class", "value": "NXentry"}],
        },
        {
            "name": "entry/data",
            "type": "Group",
            "items": [{"name": "NX_class", "value": "NXdata"}],
        },
        {
            "name": "entry/data/data",
            "type": "Dataset",
            "items": [
                {"name": "NDArrayDimBinning", "value": [1, 1]},
                {"name": "NDArrayDimOffset", "value": [0, 0]},
                {"name": "NDArrayDimReverse", "value": [0, 0]},
                {"name": "NDArrayNumDims", "value": 2},
                {"name": "signal", "value": 1},
            ],
        },
        {
            "name": "entry/instrument",
            "type": "Group",
            "items": [{"name": "NX_class", "value": "NXinstrument"}],
        },
        {
            "name": "entry/instrument/NDAttributes",
            "type": "Group",
            "items": [
                {"name": "NX_class", "value": "NXcollection"},
                {"name": "hostname", "value": "bl01c-ea-serv-01.diamond.ac.uk"},
            ],
        },
        {
            "name": "entry/instrument/NDAttributes/BayerPattern",
            "type": "Dataset",
            "items": [
                {"name": "NDAttrDescription", "value": "Bayer Pattern"},
                {"name": "NDAttrName", "value": "BayerPattern"},
                {"name": "NDAttrSource", "value": "Driver"},
                {"name": "NDAttrSourceType", "value": "NDAttrSourceDriver"},
            ],
        },
        {
            "name": "entry/instrument/NDAttributes/NDArrayEpicsTSSec",
            "type": "Dataset",
            "items": [
                {
                    "name": "NDAttrDescription",
                    "value": "The NDArray EPICS timestamp seconds past epoch",
                },
                {"name": "NDAttrName", "value": "NDArrayEpicsTSSec"},
                {"name": "NDAttrSource", "value": "Driver"},
                {"name": "NDAttrSourceType", "value": "NDAttrSourceDriver"},
            ],
        },
        {
            "name": "entry/instrument/NDAttributes/NDArrayEpicsTSnSec",
            "type": "Dataset",
            "items": [
                {
                    "name": "NDAttrDescription",
                    "value": "The NDArray EPICS timestamp nanoseconds",
                },
                {"name": "NDAttrName", "value": "NDArrayEpicsTSnSec"},
                {"name": "NDAttrSource", "value": "Driver"},
                {"name": "NDAttrSourceType", "value": "NDAttrSourceDriver"},
            ],
        },
        {
            "name": "entry/instrument/NDAttributes/NDArrayTimeStamp",
            "type": "Dataset",
            "items": [
                {
                    "name": "NDAttrDescription",
                    "value": "The timestamp of the NDArray as float64",
                },
                {"name": "NDAttrName", "value": "NDArrayTimeStamp"},
                {"name": "NDAttrSource", "value": "Driver"},
                {"name": "NDAttrSourceType", "value": "NDAttrSourceDriver"},
            ],
        },
        {
            "name": "entry/instrument/NDAttributes/NDArrayUniqueId",
            "type": "Dataset",
            "items": [
                {"name": "NDAttrDescription", "value": "The unique ID of the NDArray"},
                {"name": "NDAttrName", "value": "NDArrayUniqueId"},
                {"name": "NDAttrSource", "value": "Driver"},
                {"name": "NDAttrSourceType", "value": "NDAttrSourceDriver"},
            ],
        },
        {
            "name": "entry/instrument/detector",
            "type": "Group",
            "items": [{"name": "NX_class", "value": "NXdetector"}],
        },
        {
            "name": "entry/instrument/detector/NDAttributes",
            "type": "Group",
            "items": [{"name": "NX_class", "value": "NXcollection"}],
        },
        {
            "name": "entry/instrument/detector/NDAttributes/ColorMode",
            "type": "Dataset",
            "items": [
                {"name": "NDAttrDescription", "value": "Color Mode"},
                {"name": "NDAttrName", "value": "ColorMode"},
                {"name": "NDAttrSource", "value": "Driver"},
                {"name": "NDAttrSourceType", "value": "NDAttrSourceDriver"},
            ],
        },
        {"name": "entry/instrument/performance", "type": "Group", "items": []},
        {
            "name": "entry/instrument/performance/timestamp",
            "type": "Dataset",
            "items": [],
        },
    ]
}

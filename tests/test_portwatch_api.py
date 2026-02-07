import requests_mock

from ais_portwatch_delay_risk.portwatch_api import arcgis_query_all


def test_arcgis_query_all_pagination():
    base_url = "https://example.com/arcgis/query"
    with requests_mock.Mocker() as mocker:
        mocker.get(
            base_url,
            [
                {"json": {"features": [{"attributes": {"id": 1}}]}} ,
                {"json": {"features": [{"attributes": {"id": 2}}]}} ,
                {"json": {"features": []}},
            ],
        )
        df = arcgis_query_all(base_url, "1=1", batch_size=1)
        assert df["id"].tolist() == [1, 2]


def test_arcgis_query_all_raises_on_error():
    base_url = "https://example.com/arcgis/query"
    with requests_mock.Mocker() as mocker:
        mocker.get(
            base_url,
            json={
                "error": {
                    "code": 499,
                    "message": "Item does not exist or is inaccessible.",
                    "details": ["Item does not exist or is inaccessible."],
                }
            },
        )
        try:
            arcgis_query_all(base_url, "1=1")
        except RuntimeError as exc:
            assert "ArcGIS query failed" in str(exc)
        else:
            raise AssertionError("Expected arcgis_query_all to raise RuntimeError")

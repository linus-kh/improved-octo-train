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

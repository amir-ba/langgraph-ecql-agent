import asyncio

import httpx
import pytest

from app.tools.wfs import describe_feature_type, discover_layers, execute_wfs_query, get_layer_schema
from app.tools.wfs_client import clear_discovery_cache, filter_layers_by_subject


@pytest.fixture(autouse=True)
def _clear_discovery_cache_between_tests() -> None:
  clear_discovery_cache()


def test_discover_layers_uses_owslib_when_http_client_is_not_provided(monkeypatch) -> None:
  class _Metadata:
    def __init__(self, title: str, abstract: str) -> None:
      self.title = title
      self.abstract = abstract

  class _FakeWFS:
    def __init__(self, **_: object) -> None:
      self.contents = {
        "topp:states": _Metadata("USA States", "State polygons"),
      }

  monkeypatch.setattr("app.tools.wfs_client.WebFeatureService", _FakeWFS)

  layers = asyncio.run(discover_layers(wfs_url="https://example.test/geoserver/wfs"))

  assert layers == [
    {
      "name": "topp:states",
      "title": "USA States",
      "abstract": "State polygons",
    }
  ]


def test_describe_feature_type_uses_owslib_schema_when_http_client_is_not_provided(
  monkeypatch,
) -> None:
  class _FakeWFS:
    def __init__(self, **_: object) -> None:
      self.contents = {}

    def get_schema(self, type_name: str) -> dict[str, object]:
      assert type_name == "topp:states"
      return {
        "properties": {
          "the_geom": "gml:MultiSurfacePropertyType",
          "STATE_NAME": "xsd:string",
        },
        "geometry_column": "the_geom",
      }

  monkeypatch.setattr("app.tools.wfs_client.WebFeatureService", _FakeWFS)

  schema = asyncio.run(describe_feature_type(
    wfs_url="https://example.test/geoserver/wfs",
    type_name="topp:states",
  ))

  assert schema["geometry_column"] == "the_geom"
  assert schema["attributes"]["STATE_NAME"] == "xsd:string"


def test_discover_layers_parses_feature_types() -> None:
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <wfs:WFS_Capabilities xmlns:wfs=\"http://www.opengis.net/wfs/2.0\">
      <wfs:FeatureTypeList>
        <wfs:FeatureType>
          <wfs:Name>topp:states</wfs:Name>
          <wfs:Title>USA States</wfs:Title>
          <wfs:Abstract>State polygons</wfs:Abstract>
        </wfs:FeatureType>
      </wfs:FeatureTypeList>
    </wfs:WFS_Capabilities>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["request"] == "GetCapabilities"
        return httpx.Response(200, text=xml)

    layers = asyncio.run(discover_layers(
        wfs_url="https://example.test/geoserver/wfs",
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ))

    assert layers == [
        {
            "name": "topp:states",
            "title": "USA States",
            "abstract": "State polygons",
        }
    ]


def test_describe_feature_type_extracts_schema_and_geometry_column() -> None:
    xsd = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <xsd:schema xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\"
                xmlns:gml=\"http://www.opengis.net/gml/3.2\">
      <xsd:complexType name=\"statesType\">
        <xsd:complexContent>
          <xsd:extension>
            <xsd:sequence>
              <xsd:element name=\"the_geom\" type=\"gml:MultiSurfacePropertyType\"/>
              <xsd:element name=\"STATE_NAME\" type=\"xsd:string\"/>
              <xsd:element name=\"PERSONS\" type=\"xsd:int\"/>
            </xsd:sequence>
          </xsd:extension>
        </xsd:complexContent>
      </xsd:complexType>
    </xsd:schema>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["request"] == "DescribeFeatureType"
        assert request.url.params["typeNames"] == "topp:states"
        return httpx.Response(200, text=xsd)

    schema = asyncio.run(describe_feature_type(
        wfs_url="https://example.test/geoserver/wfs",
        type_name="topp:states",
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ))

    assert schema["geometry_column"] == "the_geom"
    assert schema["attributes"]["STATE_NAME"] == "xsd:string"
    assert schema["attributes"]["PERSONS"] == "xsd:int"


def test_execute_wfs_query_enforces_count_guardrail() -> None:
    payload = {"type": "FeatureCollection", "features": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["request"] == "GetFeature"
        assert request.url.params["typeNames"] == "topp:states"
        assert request.url.params["srsName"] == "EPSG:3857"
        assert request.url.params["cql_filter"] == "PERSONS > 1000000"
        assert request.url.params["count"] == "1000"
        return httpx.Response(200, json=payload)

    result = asyncio.run(execute_wfs_query(
        wfs_url="https://example.test/geoserver/wfs",
        type_name="topp:states",
        cql_filter="PERSONS > 1000000",
        count=5000,
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ))

    assert result == payload


def test_get_layer_schema_returns_attributes_and_geometry_column() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
            <xsd:schema xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\"
                        xmlns:gml=\"http://www.opengis.net/gml/3.2\">
              <xsd:complexType name=\"statesType\">
                <xsd:complexContent>
                  <xsd:extension>
                    <xsd:sequence>
                      <xsd:element name=\"the_geom\" type=\"gml:MultiSurfacePropertyType\"/>
                      <xsd:element name=\"STATE_NAME\" type=\"xsd:string\"/>
                    </xsd:sequence>
                  </xsd:extension>
                </xsd:complexContent>
              </xsd:complexType>
            </xsd:schema>
            """,
        )

    layer_schema, geometry_column = asyncio.run(get_layer_schema(
        wfs_url="https://example.test/geoserver/wfs",
        type_name="topp:states",
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ))

    assert layer_schema == {"the_geom": "gml:MultiSurfacePropertyType", "STATE_NAME": "xsd:string"}
    assert geometry_column == "the_geom"


def test_execute_wfs_query_uses_basic_auth_when_credentials_are_provided() -> None:
    payload = {"type": "FeatureCollection", "features": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"].startswith("Basic ")
        return httpx.Response(200, json=payload)

    result = asyncio.run(execute_wfs_query(
        wfs_url="https://example.test/geoserver/wfs",
        type_name="topp:states",
        cql_filter="PERSONS > 1000000",
        username="demo-user",
        password="demo-pass",
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ))

    assert result == payload


def test_filter_layers_by_subject_prefers_matching_name_title_and_abstract() -> None:
  layers = [
    {"name": "topp:states", "title": "USA States", "abstract": "State polygons"},
    {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"},
    {"name": "city:roads", "title": "Road Network", "abstract": "Transportation"},
  ]

  filtered = filter_layers_by_subject(layers, "hospitals")

  assert filtered == [
    {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"}
  ]


def test_discover_layers_reuses_cached_capabilities_with_same_inputs(monkeypatch) -> None:
  call_count = 0

  class _Metadata:
    def __init__(self, title: str, abstract: str) -> None:
      self.title = title
      self.abstract = abstract

  class _FakeWFS:
    def __init__(self, **_: object) -> None:
      nonlocal call_count
      call_count += 1
      self.contents = {
        "topp:states": _Metadata("USA States", "State polygons"),
      }

  monkeypatch.setattr("app.tools.wfs_client.WebFeatureService", _FakeWFS)

  first = asyncio.run(discover_layers(wfs_url="https://example.test/geoserver/wfs"))
  second = asyncio.run(discover_layers(wfs_url="https://example.test/geoserver/wfs"))

  assert first == second
  assert call_count == 1


def test_discover_layers_cache_expires_after_ttl(monkeypatch) -> None:
  call_count = 0
  clock = {"now": 1000.0}

  class _Metadata:
    def __init__(self, title: str, abstract: str) -> None:
      self.title = title
      self.abstract = abstract

  class _FakeWFS:
    def __init__(self, **_: object) -> None:
      nonlocal call_count
      call_count += 1
      self.contents = {
        "topp:states": _Metadata("USA States", "State polygons"),
      }

  monkeypatch.setattr("app.tools.wfs_client.WebFeatureService", _FakeWFS)
  monkeypatch.setattr("app.tools.wfs_client.time.time", lambda: clock["now"])

  asyncio.run(discover_layers(wfs_url="https://example.test/geoserver/wfs"))
  clock["now"] += (12 * 60 * 60) + 1
  asyncio.run(discover_layers(wfs_url="https://example.test/geoserver/wfs"))

  assert call_count == 2


def test_discover_layers_does_not_cache_failed_discovery(monkeypatch) -> None:
  call_count = 0

  class _Metadata:
    def __init__(self, title: str, abstract: str) -> None:
      self.title = title
      self.abstract = abstract

  class _FakeWFS:
    def __init__(self, **_: object) -> None:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise RuntimeError("temporary failure")
      self.contents = {
        "topp:states": _Metadata("USA States", "State polygons"),
      }

  class _FakeHttpClient:
    def __init__(self, timeout: float) -> None:
      self.timeout = timeout

    async def __aenter__(self):
      return self

    async def __aexit__(self, exc_type, exc, tb):
      return None

    async def get(self, *args, **kwargs):
      raise RuntimeError("temporary failure")

  monkeypatch.setattr("app.tools.wfs_client.WebFeatureService", _FakeWFS)
  monkeypatch.setattr("app.tools.wfs_client.httpx.AsyncClient", _FakeHttpClient)

  with pytest.raises(RuntimeError):
    asyncio.run(discover_layers(wfs_url="https://example.test/geoserver/wfs"))

  layers = asyncio.run(discover_layers(wfs_url="https://example.test/geoserver/wfs"))

  assert layers[0]["name"] == "topp:states"
  assert call_count == 2

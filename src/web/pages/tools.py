import requests
import streamlit as st


st.set_page_config(page_title="Tools", layout="centered")

st.subheader("Tools")


def get_history():
    try:
        response = requests.get("http://localhost:8000/api/history")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []


admin_token = st.text_input("Admin Token", type="password")


def delete_history(admin_token: str):
    try:
        headers = {"X-Admin-Token": admin_token}
        response = requests.delete("http://localhost:8000/api/history", headers=headers)
        response.raise_for_status()
        result = response.json()
        st.success(result["message"])
        return response.json()
    except requests.exceptions.HTTPError:
        if response.status_code == 401:
            st.error("Invalid or missing admin token.")
        else:
            st.error(f"Server error: {response.status_code} – {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")


def get_stats():
    try:
        response = requests.get("http://localhost:8000/api/stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []


if "history_data" not in st.session_state:
    st.session_state.history_data = []
if "stats_data" not in st.session_state:
    st.session_state.stats_data = []


del_history_bn = st.button("Delete history requests")

if del_history_bn:
    delete_history(admin_token)

get_history_bn = st.button("Get history requests")
get_stats_bn = st.button("Get requests stats")

if get_history_bn:
    st.session_state.history_data = get_history()


if get_stats_bn:
    st.session_state.stats_data = get_stats()

if st.session_state.history_data:
    st.divider()
    st.markdown("#### History", text_alignment="center")
    if isinstance(st.session_state.history_data, dict):
        if "message" in st.session_state.history_data.keys():
            st.info(st.session_state.history_data.get("message"))
    else:
        st.dataframe(st.session_state.history_data)

if st.session_state.stats_data:
    st.divider()
    st.markdown("#### Stats", text_alignment="center")
    if "message" in st.session_state.stats_data.keys():
        st.info(st.session_state.stats_data.get("message"))
    else:
        st.dataframe(st.session_state.stats_data)

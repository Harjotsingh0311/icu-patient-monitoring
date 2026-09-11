const API = {

    fusion: "/api/fusion",

    posture: "/api/posture",

    frame: "/api/stream/face"

};


const $ = (id) =>
    document.getElementById(id);


/* ================================================= */
/* HELPER */
/* ================================================= */

function firstDefined(
    object,
    keys,
    fallback = null
) {

    for (const key of keys) {

        if (
            object &&
            object[key] !== undefined &&
            object[key] !== null
        ) {

            return object[key];

        }

    }

    return fallback;
}


function formatNumber(
    value,
    digits = 3
) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {

        return "--";

    }

    const number = Number(value);

    if (!Number.isFinite(number)) {

        return String(value);

    }

    return number.toFixed(digits);
}


function formatPercent(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {

        return "--";

    }

    let number = Number(value);

    if (!Number.isFinite(number)) {

        return "--";

    }

    if (number <= 1) {

        number *= 100;

    }

    return `${Math.round(number)}%`;
}


/* ================================================= */
/* CONNECTION STATUS */
/* ================================================= */

function setConnection(
    state,
    text
) {

    const element =
        $("connectionStatus");

    element.className =
        `status-pill ${state}`;

    element.innerHTML =
        `<span class="dot"></span>${text}`;
}


/* ================================================= */
/* STATE */
/* ================================================= */

function updateState(face) {

    const eyeState =
        firstDefined(
            face,
            [
                "eye_state",
                "eyeState"
            ],
            "--"
        );


    const alertState =
        firstDefined(
            face,
            [
                "alert_state",
                "alertState"
            ],
            "normal"
        );


    const confidence =
        firstDefined(
            face,
            ["confidence"],
            null
        );


    $("eyeState").textContent =
        eyeState;


    $("alertState").textContent =
        String(alertState)
            .replaceAll("_", " ");


    $("confidenceValue").textContent =
        formatPercent(confidence);


    let confidenceNumber =
        Number(confidence);


    if (
        Number.isFinite(confidenceNumber)
    ) {

        if (confidenceNumber <= 1) {

            confidenceNumber *= 100;

        }


        confidenceNumber =
            Math.max(
                0,
                Math.min(
                    100,
                    confidenceNumber
                )
            );


        $("confidenceBar").style.width =
            `${confidenceNumber}%`;

    }


    const icon =
        $("stateIcon");


    icon.className =
        "state-icon";


    const state =
        String(alertState)
            .toLowerCase();


    if (
        state.includes("alert") ||
        state.includes("closure") ||
        state.includes("danger")
    ) {

        icon.classList.add(
            "danger"
        );

    }

    else if (
        state.includes("warning") ||
        state.includes("low")
    ) {

        icon.classList.add(
            "warning"
        );

    }

}


/* ================================================= */
/* METRICS */
/* ================================================= */

function updateMetrics(face) {

    $("leftEar").textContent =
        formatNumber(
            firstDefined(
                face,
                [
                    "left_ear",
                    "leftEAR",
                    "ear_left"
                ]
            )
        );


    $("rightEar").textContent =
        formatNumber(
            firstDefined(
                face,
                [
                    "right_ear",
                    "rightEAR",
                    "ear_right"
                ]
            )
        );


    $("averageEar").textContent =
        formatNumber(
            firstDefined(
                face,
                [
                    "average_ear",
                    "averageEAR",
                    "ear_avg"
                ]
            )
        );


    $("blinkCount").textContent =
        firstDefined(
            face,
            [
                "blink_count",
                "blinkCount"
            ],
            "--"
        );


    $("gazeState").textContent =
        firstDefined(
            face,
            [
                "gaze_state",
                "gazeState",
                "gaze"
            ],
            "--"
        );


    const closure =
        firstDefined(
            face,
            [
                "closure_duration",
                "closureDuration"
            ],
            null
        );


    $("closureDuration").textContent =

        closure === null

            ? "--"

            : `${formatNumber(
                closure,
                2
            )} s`;

}


/* ================================================= */
/* ALERT */
/* ================================================= */

function updateAlert(face) {

    const panel =
        $("alertPanel");


    const title =
        panel.querySelector(
            ".alert-title"
        );


    const alertState =
        String(
            firstDefined(
                face,
                [
                    "alert_state",
                    "alertState"
                ],
                "normal"
            )
        );


    const alertMessage =
        firstDefined(
            face,
            [
                "alert_message",
                "alertMessage",
                "alert",
                "message"
            ],
            "No active alert."
        );


    panel.className =
        "alert-panel";


    const lower =
        alertState.toLowerCase();


    if (
        lower.includes("alert") ||
        lower.includes("closure") ||
        lower.includes("danger")
    ) {

        panel.classList.add(
            "danger"
        );

        title.textContent =
            "ATTENTION REQUIRED";

    }

    else if (
        lower.includes("warning") ||
        lower.includes("low")
    ) {

        panel.classList.add(
            "warning"
        );

        title.textContent =
            "WARNING";

    }

    else {

        title.textContent =
            "SYSTEM NORMAL";

    }


    $("alertMessage").textContent =
        String(alertMessage);

}


/* ================================================= */
/* GET BACKEND DATA */
/* ================================================= */

async function fetchDashboardData() {

    try {

        const response =
            await fetch(
                `${API.fusion}?t=${Date.now()}`,
                {
                    cache: "no-store"
                }
            );


        if (!response.ok) {

            throw new Error(
                `Fusion HTTP ${response.status}`
            );

        }


        const data =
            await response.json();


        /*
         * Expected structure:
         *
         * {
         *     "face_eye": {...},
         *     "posture": {...}
         * }
         */


        const face =
            data.face_eye ||
            data.faceEye ||
            {};


        const posture =
            data.posture ||
            {};


        updateState(face);

        updateMetrics(face);

        updateAlert(face);


        $("postureData").textContent =
            JSON.stringify(
                posture,
                null,
                2
            );


        $("fusionData").textContent =
            JSON.stringify(
                data,
                null,
                2
            );


        $("lastUpdated").textContent =
            `Updated ${
                new Date()
                    .toLocaleTimeString()
            }`;


        setConnection(
            "connected",
            "Backend Connected"
        );

    }


    catch (error) {

        console.error(error);


        setConnection(
            "error",
            "Backend Offline"
        );

    }

}


/* ================================================= */
/* LIVE FRAME */
/* ================================================= */

function refreshFrame() {

    const image =
        $("patientFrame");


    const noFrame =
        $("noFrame");


    image.onload = () => {

        noFrame.style.display =
            "none";

    };


    image.onerror = () => {

        noFrame.style.display =
            "grid";

    };


    image.src =
        `${API.frame}?t=${Date.now()}`;

}


/* ================================================= */
/* START DASHBOARD */
/* ================================================= */

fetchDashboardData();

refreshFrame();


/*
 * Update clinical data twice per second.
 */

setInterval(
    fetchDashboardData,
    500
);


/*
 * Refresh latest camera frame
 * approximately four times per second.
 */

setInterval(
    refreshFrame,
    250
);
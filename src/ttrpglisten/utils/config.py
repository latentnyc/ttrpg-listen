"""Application configuration backed by QSettings."""

from PySide6.QtCore import QSettings

GAME_PROMPTS = {
    "Dungeons & Dragons": (
        "Dungeons & Dragons, TTRPG session. d20, armor class, hit points, initiative, "
        "saving throw, spell slot, cantrip, perception check, attack roll, natural 20, "
        "dungeon master, player character, non-player character, skill check, "
        "advantage, disadvantage, proficiency bonus."
    ),
    "Blades in the Dark": (
        "Blades in the Dark, TTRPG session. Action roll, resistance roll, position, "
        "effect, controlled, risky, desperate, stress, trauma, flashback, score, heist, "
        "downtime, engagement roll, fortune roll, devil's bargain, Cutter, Hound, "
        "Leech, Lurk, Slide, Spider, Whisper, Assassins, Bravos, Hawkers, Shadows, "
        "Smugglers, Doskvol, Duskwall, ghost field, electroplasm, Spirit Wardens, "
        "turf, rep, heat, crew, scoundrel, attune, prowl, skirmish, finesse, consort, "
        "sway, study, tinker, command, wreck, Bluecoats, Inspectors, Red Sashes, "
        "Lampblacks, Crows, Gondoliers, Ironhook Prison, Crow's Foot, Brightstone."
    ),
}


class AppConfig:
    """Persistent application configuration using QSettings."""

    def __init__(self):
        self._settings = QSettings("TTRPGListen", "TTRPGListen")

    @property
    def sample_rate(self) -> int:
        return int(self._settings.value("audio/sample_rate", 16000))

    @property
    def language(self) -> str:
        return self._settings.value("transcription/language", "en")

    @property
    def game_system(self) -> str:
        return self._settings.value("transcription/game_system", "Dungeons & Dragons")

    @game_system.setter
    def game_system(self, value: str):
        self._settings.setValue("transcription/game_system", value)
        self._settings.sync()

    @property
    def game_prompt(self) -> str:
        return GAME_PROMPTS.get(self.game_system, "")

    @property
    def min_speakers(self) -> int:
        return int(self._settings.value("diarization/min_speakers", 2))

    @property
    def max_speakers(self) -> int:
        return int(self._settings.value("diarization/max_speakers", 8))

    @property
    def transcript_directory(self) -> str:
        return self._settings.value("output/directory", "./transcripts")

    @property
    def layout_orientation(self) -> str:
        return self._settings.value("ui/layout_orientation", "horizontal")

    @layout_orientation.setter
    def layout_orientation(self, value: str):
        self._settings.setValue("ui/layout_orientation", value)
        self._settings.sync()

    @property
    def mic_gain(self) -> float:
        """Software gain multiplier applied to mic audio after resampling."""
        return float(self._settings.value("audio/mic_gain", 3.0))

    @mic_gain.setter
    def mic_gain(self, value: float):
        self._settings.setValue("audio/mic_gain", float(value))
        self._settings.sync()

    @property
    def mic_sensitivity(self) -> float:
        """RMS threshold below which mic audio is treated as ambient noise
        and excluded from the smart-mix / transcription gate."""
        return float(self._settings.value("audio/mic_sensitivity", 0.008))

    @mic_sensitivity.setter
    def mic_sensitivity(self, value: float):
        self._settings.setValue("audio/mic_sensitivity", float(value))
        self._settings.sync()

    @property
    def mic_device(self) -> int | None:
        val = self._settings.value("audio/mic_device", None)
        return int(val) if val is not None else None

    @mic_device.setter
    def mic_device(self, value: int | None):
        if value is None:
            self._settings.remove("audio/mic_device")
        else:
            self._settings.setValue("audio/mic_device", value)
        self._settings.sync()

    @property
    def loopback_device(self) -> int | None:
        val = self._settings.value("audio/loopback_device", None)
        return int(val) if val is not None else None

    @loopback_device.setter
    def loopback_device(self, value: int | None):
        if value is None:
            self._settings.remove("audio/loopback_device")
        else:
            self._settings.setValue("audio/loopback_device", value)
        self._settings.sync()

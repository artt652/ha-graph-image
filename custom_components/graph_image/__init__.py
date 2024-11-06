from homeassistant.config_entries import SOURCE_IMPORT
from homeassistant.core import HomeAssistant
from homeassistant.core import ServiceCall
from homeassistant.components.recorder import get_instance, history
import homeassistant.util.dt as dt_util

from datetime import timedelta
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

DOMAIN = 'graph_image'
_LOGGER = logging.getLogger(__name__)

executor = ThreadPoolExecutor()

async def async_setup(hass, hass_config):
    if DOMAIN in hass_config and not hass.config_entries.async_entries(DOMAIN):
        hass.async_create_task(hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_IMPORT}
        ))
    return True

async def async_setup_entry(hass: HomeAssistant, entry):

    async def save_figure_async(fig, filename):
        loop = asyncio.get_event_loop()
        # Выполняем сохранение графика в отдельном потоке
        await loop.run_in_executor(executor, fig.savefig, filename)

    async def create_subplots_async():
        loop = asyncio.get_event_loop()
        # Создаем подграфики в отдельном потоке
        return await loop.run_in_executor(executor, plt.subplots)

    async def close_figure_async(fig):
        loop = asyncio.get_event_loop()
        # Закрываем график в отдельном потоке
        await loop.run_in_executor(executor, plt.close, fig)

    async def apply_autofmt_xdate(fig):
        loop = asyncio.get_event_loop()
        # Выполняем autofmt_xdate в отдельном потоке
        await loop.run_in_executor(executor, fig.autofmt_xdate)

    async def apply_tight_layout():
        loop = asyncio.get_event_loop()
        # Выполняем tight_layout в отдельном потоке
        await loop.run_in_executor(executor, plt.tight_layout)

    # Функция для получения форматтера даты в пуле потоков
    async def get_date_formatter(tz):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            # Форматирование дат также выполняем в отдельном потоке
            xFmt = await loop.run_in_executor(pool, mdates.DateFormatter, '%m-%d %H:%M', tz)
        return xFmt

    async def create_graph_image(service: ServiceCall) -> None:
        time_start = dt_util.utcnow() - timedelta(hours=service.data.get('in_time_start', 12))  # дата начала
        time_end = dt_util.utcnow() - timedelta(hours=service.data.get('in_time_end', 0))  # дата окончания
        in_style = service.data.get('in_style', 'Solarize_Light2')  # стиль графика
        in_size_ticks = service.data.get('in_size_ticks', 10)  # размер подписи осей
        in_rate_ticks = service.data.get('in_rate_ticks', False)  # размер подписи осей
        in_linewidth = service.data.get('in_linewidth', 1)  # толщина линии
        in_folderfile = service.data.get('in_folderfile', '/config/www/graph.png')  # путь и имя для сохранения
        in_linesmooth = service.data.get('in_linesmooth', 1)  # уровень сглаживания
        in_lineinterp = service.data.get('in_lineinterp', 'linear_interp')  # вид графика
        in_size_inches = service.data.get('in_size_inches', '8,6')  # размер картинки
        entity_ids = service.data["entity_id"]

        plt.close()
        plt.style.use(in_style)  # стиль графика
        include_start_time_state = True
        no_attributes = True

        # Получаем временную зону и создаем DateFormatter в пуле потоков
        tz = hass.config.time_zone
        xFmt = await get_date_formatter(tz)

        fig, ax = await create_subplots_async()
        fig.set_size_inches(int(in_size_inches.split(',')[0]), int(in_size_inches.split(',')[1]), forward=True)  # размеры графика
        ax.margins(x=0)  # убираем все поля лишние по бокам
        plt.xticks(size=in_size_ticks)  # размер подписи по оси x
        plt.yticks(size=in_size_ticks)  # размер подписи по оси y

        # Проходимся по объектам, запрашиваем историю и формируем график
        for row in entity_ids:
            entity = hass.states.get(row)
            hist_entity = await get_instance(hass).async_add_executor_job(
                history.state_changes_during_period, hass, time_start, time_end, row, include_start_time_state, no_attributes)
            _LOGGER.debug(f"History create_graph_image {row} hist_entity: {hist_entity}")
            label_name = f"{entity.attributes.get('friendly_name', row)} - {entity.state}"
            x_axis = []
            y_axis = []
            for rec in hist_entity[row]:
                his_state = rec.state
                his_last_updated = rec.last_updated
                try:
                    float(his_state)  # проверяем, что значение числовое
                    x_axis.append(his_last_updated)
                    y_axis.append(his_state)
                except ValueError:
                    x_axis.append(his_last_updated)
                    y_axis.append(None)
            if x_axis and y_axis:
                x_axis = np.array(x_axis)
                y_axis = np.array(y_axis).astype('float32')
                w = np.hanning(in_linesmooth)  # сглаживание графика
                y_axis2 = np.convolve(w / w.sum(), y_axis, mode='same')

                if in_lineinterp == 'linear_interp':
                    ax.plot(x_axis, y_axis2, label=label_name, linewidth=in_linewidth)
                else:
                    new_where = 'pre' if in_lineinterp == 'vert_first' else 'post'
                    ax.step(x_axis, y_axis2, label=label_name, where=new_where, linewidth=in_linewidth)

        # Устанавливаем частоту делений оси X
        if not in_rate_ticks:
            diff_day = time_end - time_start
            new_interval = 1 if diff_day.days <= 1 else diff_day.days * 2
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=new_interval))

        ax.xaxis.set_major_formatter(xFmt)
        ax.legend(loc=2, ncol=2, shadow=True, fancybox=True, framealpha=0.5, fontsize=14)  # подписи сенсоров
        
        # Применяем форматирование и компоновку в отдельном потоке
        await apply_autofmt_xdate(fig)
        await apply_tight_layout()

        # Save figure asynchronously
        await save_figure_async(fig, in_folderfile)

        # Close figure asynchronously
        await close_figure_async(fig)

    hass.services.async_register(DOMAIN, 'create_graph_image', create_graph_image)

    return True

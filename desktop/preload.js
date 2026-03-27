const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  draft: {
    save:  (data) => ipcRenderer.invoke('draft:save', data),
    load:  ()     => ipcRenderer.invoke('draft:load'),
    clear: ()     => ipcRenderer.invoke('draft:clear'),
  },
  power: {
    keepAwake:  () => ipcRenderer.invoke('power:keepAwake'),
    allowSleep: () => ipcRenderer.invoke('power:allowSleep'),
  },
  desktop: {
    getSources: (opts) => ipcRenderer.invoke('desktop:getSources', opts),
  },
});
